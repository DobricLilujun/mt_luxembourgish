import argparse
import os
import warnings
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import time
import torch
import sys
from datasets import Dataset

# ========================== CMD Argument Parser ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with Hugging Face Transformers")
    parser.add_argument("--per_device_train_batch_size", type=int, default=10, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=10, help="Batch size per device during evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add_argument("--training_dataset_path", type=str, default="data/processed/dataset_merged_llama_fake_targets.jsonl", help="Path to training dataset")
    return parser.parse_args()


# ========================== Main Training Code ==========================
def train_ddp_accelerate(args):
    project_root = "/Users/lujun.li/projects/mt_luxembourgish"
    model_name = "/home/snt/llm_models/Llama-3.2-1B-Instruct"
    # "/home/llama/Personal_Directories/srb/binary_classfication/Llama-3.2-3B-Instruct"

    train_dataset_path = os.path.abspath(os.path.join(project_root, args.training_dataset_path))
    val_dataset_path = os.path.abspath(os.path.join(project_root, "data/fake_targets/flores_devtest_arrow"))

    # Load dataset
    if train_dataset_path.endswith(".jsonl"):
        train_dataset = Dataset.from_json(train_dataset_path)
    else:
        train_dataset = load_from_disk(train_dataset_path)

    train_dataset = train_dataset.rename_columns({
        "input": "Luxembourgish",
        "translated_text": "English",
    })

    val_dataset = (
        load_from_disk(val_dataset_path)
        .rename_columns(
            {
                "sentence_ltz_Latn": "Luxembourgish",
                "sentence_eng_Latn": "English",
            }
        )
        .select([i for i in range(100)])
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def create_prompt(sample, mode="train"):
        system_message = f"Translate the Luxembourgish input text into English.".upper()
        input_text = sample["Luxembourgish"].strip()
        response = sample["English"].strip() if "English" in sample else ""
        eos_token = tokenizer.eos_token
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        full_prompt += f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        if mode == "train":
            full_prompt += response + eos_token
        return {"prompt_response": full_prompt}

    train_dataset = train_dataset.map(lambda sample: create_prompt(sample)).select_columns(["prompt_response"])
    val_dataset = val_dataset.map(lambda sample: create_prompt(sample)).select_columns(["prompt_response"])

    dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

    def tokenize_function(examples):
        return tokenizer(
            examples["prompt_response"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["prompt_response"])
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["prompt_response"])

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.use_cache = False

    current = time.time()
    output_dir = f"logs/fit_{current}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        fp16=True,
        max_grad_norm=0.3,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        disable_tqdm=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    train_ddp_accelerate(args)


# python notebook/training/CPT.py --per_device_train_batch_size 10 --per_device_eval_batch_size 10 --num_train_epochs 5 --learning_rate 1e-6 --training_dataset_path data/processed/dataset_merged_llama_fake_targets.jsonl
