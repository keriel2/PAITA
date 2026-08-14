import argparse
import gc
import json
import logging
import os
import sys

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from data import (
    PAITADataCollator,
    create_eval_dataset_with_template_ids,
    prepare_training_data,
    sample_balanced_dataset,
)
from evaluation import (
    EnhancedTrainer,
    EnsureTrainingContinuesCallback,
    FinalModelOnlyCallback,
    FixedEarlyStoppingCallback,
    PAITAPerformanceCallback,
    compute_per_template_metrics,
)
from prompt_managers import PROMPT_MANAGERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train with a given instruction-template assignment strategy.")
    parser.add_argument("--strategy", choices=list(PROMPT_MANAGERS.keys()), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples_per_task", type=int, default=9500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=48)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--eval_max_samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model_and_tokenizer(model_path, local_rank):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", truncation_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    device_map = {"": local_rank} if torch.cuda.device_count() > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        use_cache=False,
        device_map=device_map,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer


def apply_lora(model, r, alpha):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="lora_only",
        modules_to_save=["norm", "embed_tokens"],
    )
    return get_peft_model(model, lora_config)


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading model and tokenizer from {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, local_rank)
    model = apply_lora(model, args.lora_r, args.lora_alpha)

    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    balanced_df = sample_balanced_dataset(df, samples_per_task=args.samples_per_task, random_seed=args.seed)
    train_dataset, val_dataset = prepare_training_data(balanced_df, random_seed=args.seed)

    prompt_manager_cls = PROMPT_MANAGERS[args.strategy]
    prompt_manager = prompt_manager_cls(tokenizer=tokenizer, max_length=args.max_length)
    logger.info(f"Using strategy: {args.strategy} ({prompt_manager_cls.__name__})")

    val_dataset_small = create_eval_dataset_with_template_ids(
        val_dataset, prompt_manager, max_samples=args.eval_max_samples
    )

    collator = PAITADataCollator(tokenizer=tokenizer, prompt_manager=prompt_manager, max_length=args.max_length)

    def compute_metrics(eval_preds):
        return compute_per_template_metrics(eval_preds, val_dataset_small, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=f"paita-{args.strategy}",
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        optim="adamw_8bit",
        logging_steps=100,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        save_total_limit=0,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        save_strategy="no",
        evaluation_strategy="epoch",
        lr_scheduler_type="cosine_with_restarts",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=0.5,
        warmup_ratio=0.03,
        local_rank=local_rank,
    )

    callbacks = [
        FixedEarlyStoppingCallback(early_stopping_patience=10),
        FinalModelOnlyCallback(),
        EnsureTrainingContinuesCallback(),
        PAITAPerformanceCallback(prompt_manager, val_dataset_small),
    ]

    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_small,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    trainer.train()

    is_main_process = local_rank in (-1, 0)
    if is_main_process:
        final_model_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        stats_path = os.path.join(args.output_dir, "prompt_strategy_stats.json")
        with open(stats_path, "w") as f:
            json.dump(prompt_manager.get_stats(), f, indent=2)
        logger.info(f"Strategy stats saved to {stats_path}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
