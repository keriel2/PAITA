import json
import logging
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


class EnhancedTrainer(Trainer):
    def training_step(self, model, inputs):
        inputs.pop("template_ids", None)
        return super().training_step(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs.pop("template_ids", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class FixedEarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=1, early_stopping_threshold=0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        metric_to_check = args.metric_for_best_model
        if not metric_to_check or metric_to_check not in metrics:
            return
        metric_value = metrics[metric_to_check]
        greater_is_better = args.greater_is_better

        if self.best_metric is None:
            self.best_metric = metric_value
            self.early_stopping_patience_counter = 0
        elif greater_is_better and metric_value >= (self.best_metric + self.early_stopping_threshold):
            self.best_metric = metric_value
            self.early_stopping_patience_counter = 0
        elif not greater_is_better and metric_value <= (self.best_metric - self.early_stopping_threshold):
            self.best_metric = metric_value
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                control.should_training_stop = True

        return control


class FinalModelOnlyCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        control.should_save = False
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = False
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        final_model_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        trainer = kwargs.get("trainer")
        if trainer:
            trainer.save_model(final_model_dir)
            if hasattr(trainer, "tokenizer") and trainer.tokenizer:
                trainer.tokenizer.save_pretrained(final_model_dir)
            torch.save(args, os.path.join(final_model_dir, "training_args.bin"))
            if getattr(state, "best_metric", None) is not None:
                with open(os.path.join(final_model_dir, "final_metrics.json"), "w") as f:
                    json.dump({
                        "final_step": state.global_step,
                        "final_epoch": state.epoch,
                        "best_metric": state.best_metric,
                        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }, f, indent=2)
        return control


class EnsureTrainingContinuesCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        control.should_training_stop = False
        return control


def compute_per_template_metrics(eval_preds, eval_dataset, tokenizer):
    if hasattr(eval_preds, "predictions") and hasattr(eval_preds, "label_ids"):
        predictions, labels = eval_preds.predictions, eval_preds.label_ids
    elif isinstance(eval_preds, tuple) and len(eval_preds) >= 2:
        predictions, labels = eval_preds[0], eval_preds[1]
    else:
        return {"eval_loss": 0.0}

    if isinstance(predictions, np.ndarray) and predictions.ndim > 2:
        predictions = predictions.argmax(axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels_cleaned = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_cleaned, skip_special_tokens=True)

    template_ids = eval_dataset["template_id"]
    tasks = eval_dataset["task"]

    grouped = defaultdict(lambda: {"preds": [], "refs": []})
    for pred, ref, task, template_id in zip(decoded_preds, decoded_labels, tasks, template_ids):
        grouped[(task, template_id)]["preds"].append(pred)
        grouped[(task, template_id)]["refs"].append(ref)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    template_metrics = defaultdict(dict)
    all_metrics = {}

    for (task, template_id), data in grouped.items():
        preds, refs = data["preds"], data["refs"]

        if task == "Clinical Diagnosis":
            matches = sum(1 for p, r in zip(preds, refs) if p.strip().lower() == r.strip().lower())
            score = matches / len(preds) if preds else 0.0
            template_metrics[task].setdefault(template_id, {})["diagnosis_accuracy"] = score
        elif task == "Medical Question Answering":
            scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs) if p.strip() and r.strip()]
            score = sum(scores) / len(scores) if scores else 0.0
            template_metrics[task].setdefault(template_id, {})["rougeL"] = score
        elif task == "Clinical Reasoning":
            conclusion_scores = []
            for p, r in zip(preds, refs):
                p_tokens, r_tokens = set(p.lower().split()), set(r.lower().split())
                if r_tokens:
                    conclusion_scores.append(len(p_tokens & r_tokens) / len(r_tokens))
            score = sum(conclusion_scores) / len(conclusion_scores) if conclusion_scores else 0.0
            template_metrics[task].setdefault(template_id, {})["conclusion_accuracy"] = score
        else:
            continue

        for metric_name, metric_value in template_metrics[task][template_id].items():
            all_metrics[f"{task}_template_{template_id}_{metric_name}"] = metric_value

    all_metrics["_template_metrics"] = dict(template_metrics)
    all_scores = [v for t in template_metrics.values() for d in t.values() for v in d.values()]
    all_metrics["eval_rougeL"] = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return all_metrics


class PAITAPerformanceCallback(TrainerCallback):
    def __init__(self, prompt_manager, eval_dataset):
        self.prompt_manager = prompt_manager
        self.eval_dataset = eval_dataset
        self.evaluation_count = 0

    def _is_main_process(self):
        try:
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            return local_rank in (-1, 0)
        except Exception:
            return True

    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info(f"Epoch {state.epoch:.1f} starting")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self._is_main_process() or not metrics:
            return control

        self.evaluation_count += 1
        template_metrics = metrics.get("_template_metrics", {})
        if not template_metrics:
            return control

        for task, template_scores in template_metrics.items():
            if task not in self.prompt_manager.template_performance:
                continue
            for template_id, scores in template_scores.items():
                self.prompt_manager.update_template_performance(task, template_id, scores)

        self._save_stats(args, state)
        return control

    def _save_stats(self, args, state):
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        try:
            stats = self.prompt_manager.get_stats()
            stats["_metadata"] = {"evaluation_step": int(state.global_step), "epoch": float(state.epoch)}
            strategy = self.prompt_manager.strategy_name
            path = os.path.join(output_dir, f"{strategy}_stats_step_{state.global_step}.json")
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Save failed: {e}")
