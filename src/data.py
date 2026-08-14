import logging
import random

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def sample_balanced_dataset(df, samples_per_task=9500, random_seed=42):
    random.seed(random_seed)

    tasks = df["task"].unique()
    logger.info(f"Found {len(tasks)} unique tasks")

    sampled_data = []
    for task in tasks:
        task_df = df[df["task"] == task]
        if len(task_df) > samples_per_task:
            task_sample = task_df.sample(samples_per_task, random_state=random_seed)
        else:
            task_sample = task_df
        sampled_data.append(task_sample)

    balanced_df = pd.concat(sampled_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    logger.info(f"Final balanced dataset has {len(balanced_df)} examples")
    return balanced_df


def prepare_training_data(df, test_size=0.1, random_seed=42):
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe")

    required_columns = ["task", "input", "output"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df["task"], random_state=random_seed)

    train_df = train_df[required_columns].copy().reset_index(drop=True)
    val_df = val_df[required_columns].copy().reset_index(drop=True)
    train_df.fillna("", inplace=True)
    val_df.fillna("", inplace=True)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False).flatten_indices()
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False).flatten_indices()

    logger.info(f"Datasets ready - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def create_eval_dataset_with_template_ids(eval_dataset, prompt_manager, max_samples=60):
    tasks = eval_dataset["task"]
    unique_tasks = list(set(tasks))
    samples_per_task = max_samples // len(unique_tasks)

    selected_examples = []
    for task in unique_tasks:
        task_indices = [i for i, t in enumerate(tasks) if t == task]
        selected_indices = task_indices[:samples_per_task]
        num_templates = len(prompt_manager.templates[task]["templates"])

        for i, idx in enumerate(selected_indices):
            example = eval_dataset[idx]
            template_id = i % num_templates
            selected_examples.append({
                "task": example["task"],
                "input": example["input"],
                "output": example["output"],
                "template_id": template_id,
            })

    eval_dataset_with_ids = Dataset.from_list(selected_examples)
    logger.info(f"Created eval dataset with template IDs: {len(eval_dataset_with_ids)} examples")
    return eval_dataset_with_ids


class PAITADataCollator:
    def __init__(self, tokenizer, prompt_manager, max_length=512):
        self.tokenizer = tokenizer
        self.prompt_manager = prompt_manager
        self.max_length = max_length
        self.batch_count = 0
        logger.info(f"DataCollator initialized with {prompt_manager.strategy_name} strategy")

    def __call__(self, features):
        self.batch_count += 1
        if not features:
            return self._create_dummy_batch()

        try:
            formatted_prompts = []
            completed_texts = []

            for feature in features:
                task = feature.get("task", "Unknown")
                input_text = feature.get("input", "")
                output_text = feature.get("output", "")
                template_id = feature.get("template_id")

                if template_id is not None and template_id >= 0:
                    formatted_prompt = self.prompt_manager.format_with_template_id(task, input_text, template_id)
                else:
                    formatted_prompt = self.prompt_manager.format_prompt(task, input_text, for_eval=False)

                formatted_prompts.append(formatted_prompt)

                if output_text:
                    output_markers = ["Output:", "Diagnosis:", "Answer:", "Reasoning:", "Response:", "Treatment:"]
                    split_pos = -1
                    for marker in output_markers:
                        if marker in formatted_prompt:
                            split_pos = formatted_prompt.find(marker) + len(marker)
                            break
                    if split_pos != -1:
                        completed_text = formatted_prompt[:split_pos] + " " + output_text
                    else:
                        completed_text = formatted_prompt + " " + output_text
                else:
                    completed_text = formatted_prompt

                completed_texts.append(completed_text)

            model_inputs = self.tokenizer(
                completed_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            labels = model_inputs["input_ids"].clone()
            for i, prompt in enumerate(formatted_prompts):
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                prompt_length = len(prompt_tokens)
                if prompt_length < self.max_length:
                    labels[i, :prompt_length] = -100
                padding_mask = model_inputs["attention_mask"][i] == 0
                labels[i, padding_mask] = -100

            model_inputs["labels"] = labels

            if self.batch_count == 1:
                logger.info(f"First batch formatted with {self.prompt_manager.strategy_name} strategy")
                logger.info(f"Sample prompt: {formatted_prompts[0][:200]}")

            return model_inputs

        except Exception as e:
            logger.error(f"Error in collator: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_dummy_batch()

    def _create_dummy_batch(self):
        dummy_text = "This is a dummy training example."
        model_inputs = self.tokenizer(
            [dummy_text], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = model_inputs["input_ids"].clone()
        labels[:, :] = -100
        model_inputs["labels"] = labels
        return model_inputs
