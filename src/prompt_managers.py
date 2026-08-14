import random
from collections import defaultdict, Counter

import numpy as np
import torch

from templates import TEMPLATE_POOLS


class BasePromptManager:
    strategy_name = "base"

    def __init__(self, tokenizer=None, max_length=512, debug=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = debug
        self.templates = {task: {"templates": list(pool)} for task, pool in TEMPLATE_POOLS.items()}

    def format_with_template_id(self, task, input_text, template_id):
        if task not in self.templates:
            return f"Input: {input_text}\nOutput:"
        pool = self.templates[task]["templates"]
        idx = int(template_id) % len(pool)
        return pool[idx].format(input_text=input_text)


class StaticPromptManager(BasePromptManager):
    strategy_name = "static"

    def __init__(self, tokenizer=None, max_length=512, debug=False):
        super().__init__(tokenizer, max_length, debug)
        self.format_count = 0
        self.task_usage = defaultdict(int)

    def format_prompt(self, task: str, input_text: str, for_eval: bool = False) -> str:
        self.format_count += 1
        self.task_usage[task] += 1
        if task not in self.templates:
            return f"Input: {input_text}\nOutput:"
        return self.templates[task]["templates"][0].format(input_text=input_text)

    def get_stats(self):
        return {
            "strategy": self.strategy_name,
            "total_prompts_formatted": self.format_count,
            "task_usage": dict(self.task_usage),
        }


class RandomPromptManager(BasePromptManager):
    strategy_name = "random"

    def __init__(self, tokenizer=None, max_length=512, debug=False):
        super().__init__(tokenizer, max_length, debug)
        self.format_count = 0
        self.template_usage = defaultdict(int)
        self.selection_history = []
        self.last_selected_templates = []
        self.template_selection_counts = defaultdict(int)
        self.template_performance = {
            task: {i: 0.5 for i in range(len(pool["templates"]))}
            for task, pool in self.templates.items()
        }

    def format_prompt(self, task: str, input_text: str, for_eval: bool = False) -> str:
        if task not in self.templates or not self.templates[task]["templates"]:
            return f"Input: {input_text}\nOutput:"

        templates = self.templates[task]["templates"]
        selected_idx = random.randint(0, len(templates) - 1)
        selected_template = templates[selected_idx]

        template_key = f"{task}_template_{selected_idx}"
        self.selection_history.append({
            "step": self.format_count,
            "task": task,
            "template_idx": selected_idx,
            "template_key": template_key,
            "selection_type": "random",
            "for_eval": for_eval,
        })
        self.last_selected_templates.append(template_key)
        if len(self.last_selected_templates) > 100:
            self.last_selected_templates = self.last_selected_templates[-100:]
        self.template_selection_counts[template_key] += 1
        self.template_usage[template_key] += 1
        self.format_count += 1

        return selected_template.format(input_text=input_text)

    def update_template_performance(self, task, template_idx, performance_metrics):
        if task not in self.template_performance:
            return
        current_score = self.template_performance[task].get(template_idx, 0.5)
        metric_values = [v for v in performance_metrics.values() if isinstance(v, (int, float)) and v > 0]
        if metric_values:
            new_score = sum(metric_values) / len(metric_values)
            learning_rate = 0.15
            updated = current_score * (1 - learning_rate) + new_score * learning_rate
            self.template_performance[task][template_idx] = min(max(updated, 0.0), 1.0)

    def get_stats(self):
        stats = {
            "strategy": self.strategy_name,
            "total_prompts_formatted": self.format_count,
            "total_selections": len(self.selection_history),
            "template_usage": dict(self.template_usage),
            "template_selection_counts": dict(self.template_selection_counts),
            "template_performance": {t: dict(s) for t, s in self.template_performance.items()},
        }
        if self.last_selected_templates:
            stats["recent_template_preferences"] = dict(Counter(self.last_selected_templates).most_common(10))
        return stats


class PerformanceAwareLearningManager(BasePromptManager):
    strategy_name = "paita"

    def __init__(self, tokenizer=None, max_length=512, debug=False, track_only_task=None):
        super().__init__(tokenizer, max_length, debug)
        self.track_only_task = track_only_task

        self.template_performance = {
            task: {i: 0.5 for i in range(len(pool["templates"]))}
            for task, pool in self.templates.items()
        }

        self.ucb_c = 1.0
        self.learning_rate = 0.15

        self.warmup_min_rounds_per_template = 20
        total_templates = sum(len(pool["templates"]) for pool in self.templates.values())
        self.warmup_prompts = total_templates * self.warmup_min_rounds_per_template

        self.warmup_index = {task: 0 for task in self.templates}
        self.warmup_usage_count = {
            task: {i: 0 for i in range(len(pool["templates"]))}
            for task, pool in self.templates.items()
        }

        self.template_usage = defaultdict(int)
        self.selection_history = []
        self.last_selected_templates = []
        self.template_selection_counts = defaultdict(int)
        self.format_count = 0

    def _is_warmup_complete(self):
        if self.format_count >= self.warmup_prompts:
            for task in self.warmup_usage_count:
                for count in self.warmup_usage_count[task].values():
                    if count < self.warmup_min_rounds_per_template:
                        return False
            return True
        return False

    def _select_template_warmup(self, task):
        num_templates = len(self.templates[task]["templates"])
        template_id = self.warmup_index[task] % num_templates
        self.warmup_index[task] += 1
        self.warmup_usage_count[task][template_id] += 1
        return template_id, "warmup_round_robin"

    def _select_template_ucb(self, task, templates):
        scores = self.template_performance[task]
        total_usage = sum(self.template_usage.get(f"{task}_template_{i}", 1) for i in range(len(templates)))
        total_usage = max(total_usage, len(templates))

        ucb_scores = {}
        for idx in range(len(templates)):
            key = f"{task}_template_{idx}"
            usage = max(self.template_usage.get(key, 0), 1)
            avg_score = scores.get(idx, 0.5)
            exploration_bonus = self.ucb_c * np.sqrt(np.log(total_usage) / usage)
            ucb_scores[idx] = avg_score + exploration_bonus

        best = max(ucb_scores.items(), key=lambda x: x[1])
        return best[0], "ucb"

    def format_prompt(self, task: str, input_text: str, for_eval: bool = False) -> str:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        if task not in self.templates:
            return f"Input: {input_text}\nOutput:"

        templates = self.templates[task]["templates"]

        if rank == 0:
            if not self._is_warmup_complete():
                selected_idx, selection_type = self._select_template_warmup(task)
            else:
                selected_idx, selection_type = self._select_template_ucb(task, templates)
            selected_tensor = torch.tensor([selected_idx], dtype=torch.long)
            if torch.cuda.is_available():
                selected_tensor = selected_tensor.cuda()
        else:
            selected_tensor = torch.zeros(1, dtype=torch.long)
            if torch.cuda.is_available():
                selected_tensor = selected_tensor.cuda()

        if world_size > 1:
            torch.distributed.broadcast(selected_tensor, src=0)

        selected_idx = selected_tensor.item()
        selected_template = templates[selected_idx]

        if rank == 0:
            template_id = f"{task}_template_{selected_idx}"
            self.template_usage[template_id] += 1
            self.template_selection_counts[template_id] += 1
            self.format_count += 1
            self.selection_history.append({
                "step": self.format_count,
                "task": task,
                "template_idx": selected_idx,
                "selection_type": selection_type,
                "in_warmup": not self._is_warmup_complete(),
                "rank": rank,
            })

        return selected_template.format(input_text=input_text)

    def update_template_performance(self, task, template_idx, performance_metrics):
        if task not in self.template_performance:
            return

        if task == "Clinical Diagnosis":
            new_score = performance_metrics.get("diagnosis_accuracy") or performance_metrics.get("accuracy") or 0.0
        elif task == "Clinical Reasoning":
            new_score = performance_metrics.get("conclusion_accuracy", 0.0)
        else:
            new_score = performance_metrics.get("rougeL", 0.0)

        if new_score == 0.0:
            values = [v for v in performance_metrics.values() if isinstance(v, (int, float)) and v > 0]
            new_score = sum(values) / len(values) if values else 0.5

        current = self.template_performance[task].get(template_idx, 0.5)
        updated = (1 - self.learning_rate) * current + self.learning_rate * new_score
        self.template_performance[task][template_idx] = min(max(updated, 0.0), 1.0)

    def get_stats(self):
        stats = {
            "strategy": self.strategy_name,
            "selection_algorithm": "UCB",
            "ucb_c": self.ucb_c,
            "tracked_task": self.track_only_task or "all tasks",
            "total_prompts_formatted": self.format_count,
            "warmup_prompts": self.warmup_prompts,
            "total_selections": len(self.selection_history),
            "learning_rate": self.learning_rate,
        }
        if self.selection_history:
            warmup_count = sum(1 for s in self.selection_history if s["in_warmup"])
            ucb_count = sum(1 for s in self.selection_history if s["selection_type"] == "ucb")
            stats["warmup_selections"] = warmup_count
            stats["ucb_selections"] = ucb_count
            stats["warmup_ratio"] = warmup_count / len(self.selection_history)

        stats["template_usage"] = dict(self.template_usage)
        stats["template_selection_counts"] = dict(self.template_selection_counts)
        stats["template_performance"] = {t: dict(s) for t, s in self.template_performance.items()}

        if self.last_selected_templates:
            stats["recent_template_preferences"] = dict(Counter(self.last_selected_templates).most_common(10))

        return stats


PROMPT_MANAGERS = {
    "static": StaticPromptManager,
    "random": RandomPromptManager,
    "paita": PerformanceAwareLearningManager,
}
