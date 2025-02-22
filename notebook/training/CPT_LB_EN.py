# ====================================================IMPORT=================================================================

from datasets import DatasetDict, load_dataset, load_from_disk, Dataset

import os
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import time
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
os.environ.update(
    {
        # "NCCL_P2P_DISABLE": "1",
        # "NCCL_IB_DISABLE": "1",
        # "TOKENIZERS_PARALLELISM": "false",
        # "CUDA_VISIBLE_DEVICES": "3,2,1",
    }
)

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import pandas as pd
import sys
import os
from datasets import Dataset
import argparse


# ========================== CMD Argument Parser ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using CPT (Continual Pretraining Training)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=10, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=10, help="Batch size per device during evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add_argument("--project_root", type=str, default="/Users/lujun.li/projects/mt_luxembourgish", help="Path to project root")
    parser.add_argument("--training_dataset_path", type=str, default="data/processed/dataset_merged_llama_fake_targets.jsonl", help="Path to training dataset")
    parser.add_argument("--model_name", type=str, default="/home/llama/Personal_Directories/srb/binary_classfication/Llama-3.2-3B-Instruct", help="Path to model")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False, help="Resume training from checkpoint")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()

args = parse_args()

print("Arguments passed:")
print(f"Train Batch Size: {args.per_device_train_batch_size}")
print(f"Eval Batch Size: {args.per_device_eval_batch_size}")
print(f"Number of Epochs: {args.num_train_epochs}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Project Root: {args.project_root}")
print(f"Training Dataset Path: {args.training_dataset_path}")
print(f"Model Name: {args.model_name}")

learning_rate = args.learning_rate # Learning rate for the optimizer
per_device_train_batch_size = args.per_device_train_batch_size  # Batch size for training per device
per_device_eval_batch_size = args.per_device_eval_batch_size  # Batch size for evaluation per device
num_train_epochs = args.num_train_epochs  # Number of epochs for training
training_dataset_path = args.training_dataset_path
project_root = args.project_root
model_name = args.model_name
resume_from_checkpoint = args.resume_from_checkpoint
resume_checkpoint_path = args.resume_checkpoint_path

if resume_from_checkpoint and resume_checkpoint_path is None:
    raise ValueError("Please provide a checkpoint path to resume training from")


train_ratio = 1.0  # Number of samples to be used for training and evaluation
weight_decay = 0.01  # Weight decay rate for regularization
MAX_LEN = 512  # Maximum sequence length for model inputs
warmup_ratio = 0.5
logging_steps = 300
evaluation_strategy="epoch"
save_strategy="epoch"
max_grad_norm = 0.3
fp16 = True
resume_from_checkpoint = resume_from_checkpoint

# ========================== Main Training Code ==========================

val_dataset_path = os.path.abspath(os.path.join(project_root, "data/fake_targets/flores_devtest_arrow"))
train_dataset_path = os.path.abspath(os.path.join(project_root, training_dataset_path))
sys.path.append(project_root)

# Load dataset
if train_dataset_path.endswith(".jsonl"):
    dataset = Dataset.from_json(train_dataset_path)  # Ensure correct format
else:
    dataset = load_from_disk(train_dataset_path)

# Filter by split
train_dataset = dataset.filter(lambda x: x["split"] == "train")
val_dataset = dataset.filter(lambda x: x["split"] == "val")


# Select subset
train_dataset = train_dataset.select(range(int(len(train_dataset) * train_ratio)))
val_dataset = val_dataset.select(range(int(len(val_dataset) * train_ratio)))  # Avoid out-of-range error

# Rename columns
train_dataset = train_dataset.rename_columns({
    "input": "Luxembourgish",
    "translated_text": "English",
})

val_dataset = val_dataset.rename_columns({
    "input": "Luxembourgish",
    "translated_text": "English",
})


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def create_prompt(
    sample, mode="train", src_lng="Luxembourgish", tgt_lng="English", tokenizer=None
):
    """
    Create a prompt using the model's EOS token.

    Args:
        sample (dict): A dictionary containing source and target text.
        mode (str): The mode, either 'train' or 'test'.
        src_lng (str): Source language name.
        tgt_lng (str): Target language name.
        tokenizer: The tokenizer associated with the model (required to fetch EOS token).

    Returns:
        dict: A dictionary with the constructed prompt.
    """
    # Validate the tokenizer input
    if tokenizer is None or tokenizer.eos_token is None:
        raise ValueError("A tokenizer with a defined EOS token is required.")

    # Define the system message template.
    system_message = f"Translate the {src_lng} input text into {tgt_lng}.".upper()
    input_text = sample[src_lng.capitalize()].strip()  # Extract the input text.
    response = (
        sample[tgt_lng.capitalize()].strip() if tgt_lng.capitalize() in sample else ""
    )  # Extract the target text.

    # Get the EOS token from the tokenizer.
    eos_token = tokenizer.eos_token

    # Construct the full prompt.
    full_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + system_message
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )
    full_prompt += (
        input_text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if mode == "train":
        full_prompt += response + eos_token
    return {"prompt_response": full_prompt}


train_dataset = train_dataset.map(
    lambda sample: {
        "prompt_response": create_prompt(sample, mode="train", tokenizer=tokenizer)[
            "prompt_response"
        ]
    }
).select_columns(["prompt_response"])

val_dataset = val_dataset.map(
    lambda sample: {
        "prompt_response": create_prompt(sample, mode="train", tokenizer=tokenizer)[
            "prompt_response"
        ]
    }
).select_columns(["prompt_response"])


dataset = DatasetDict({"train": train_dataset, "val": val_dataset})
data_collator = DataCollatorForLanguageModeling(
    tokenizer, mlm=False, return_tensors="pt"
)


def tokenize_function(examples):
    return tokenizer(
        examples["prompt_response"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )


tokenized_train_dataset = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["prompt_response"]
)
tokenized_val_dataset = val_dataset.map(
    tokenize_function, batched=True, remove_columns=["prompt_response"]
)

# ====================================================TRAINING=================================================================
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"

def train_ddp_accelerate():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.use_cache = False
    current = time.time()
    if resume_from_checkpoint:
        output_dir = resume_checkpoint_path
    else:
        input_file_name = training_dataset_path.split("/")[-1].split(".")[0]
        output_dir = f"logs/fit_{current}_{train_ratio}_{input_file_name}"

    print(print_trainable_parameters(model))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_ratio=warmup_ratio,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        # save_steps=save_steps,
        # eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
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
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Finished training.")
    

from accelerate import Accelerator


def main():
    accelerator = Accelerator()
    train_ddp_accelerate()

if __name__ == "__main__":
    main()

# python notebook/training/CPT.py --per_device_train_batch_size 10 --per_device_eval_batch_size 10 --num_train_epochs 5 --learning_rate 1e-6 --training_dataset_path data/processed/dataset_merged_llama_fake_targets.jsonl

# python notebook/training/CPT.py \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --num_train_epochs 3 \
#     --learning_rate 1e-6 \
#     --project_root "/home/snt/projects_lujun/mt_luxembourgish" \
#     --training_dataset_path "data/processed/dataset_merged_llama_fake_targets_with_split.jsonl" \
#     --model_name "/home/snt/projects_lujun/base_models/Llama-3.2-1B-Instruct" \
#     --resume_from_checkpoint True\
#     --resume_checkpoint_path "/home/snt/projects_lujun/mt_luxembourgish/logs/fit_1738867685.359803_0.001"



# python notebook/training/CPT.py \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --num_train_epochs 3 \
#     --learning_rate 1e-6 \
#     --project_root "/home/snt/projects_lujun/mt_luxembourgish" \
#     --training_dataset_path "data/processed/dataset_merged_llama_fake_targets_with_split.jsonl" \
#     --model_name "/home/snt/projects_lujun/base_models/Llama-3.2-3B-Instruct"