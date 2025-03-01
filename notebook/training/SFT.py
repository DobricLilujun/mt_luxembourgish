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

from utils.utils_train import pre_process, create_prompt, print_trainable_parameters
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


# ========================== CMD Argument Parser ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using CPT (Continual Pretraining Training)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation")
    parser.add_argument("--src_lng", type=str, default="English", help="Source language default English")
    parser.add_argument("--tgt_lng", type=str, default="Luxembourgish", help="Target language default Luxembourgish")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training")
    parser.add_argument("--project_root", type=str, default="/Users/lujun.li/projects/mt_luxembourgish", help="Path to project root")
    parser.add_argument("--training_dataset_path", type=str, default="data/processed/dataset_merged_llama_fake_targets.jsonl", help="Path to training dataset")
    parser.add_argument("--model_path", type=str, default="/home/llama/Personal_Directories/srb/binary_classfication/Llama-3.2-3B-Instruct", help="Path to model")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False, help="Resume training from checkpoint")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--r", type=int, default=256, help="Number of random samples to be used for training")
    parser.add_argument("--is_peft", type=bool, default=False, help="Use PEFT")
    parser.add_argument("--is_unsloth", type=bool, default=False, help="Use UnSloth")
    parser.add_argument("--is_train_response_only", type=bool, default=True, help="Train response only")
    return parser.parse_args()

args = parse_args()

print("Arguments passed:")
print(f"Train Batch Size: {args.per_device_train_batch_size}")
print(f"Eval Batch Size: {args.per_device_eval_batch_size}")
print(f"Number of Epochs: {args.num_train_epochs}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Project Root: {args.project_root}")
print(f"Training Dataset Path: {args.training_dataset_path}")
print(f"Model path: {args.model_path}")
print(f"tgt_lng: {args.tgt_lng}")
print(f"src_lng: {args.src_lng}")
print(f"Resume from checkpoint: {args.resume_from_checkpoint}")
print(f"Resume checkpoint path: {args.resume_checkpoint_path}")
print(f"r: {args.r}")
print(f"is_peft: {args.is_peft}")
print(f"is_unsloth: {args.is_unsloth}")
print(f"is_train_response_only: {args.is_train_response_only}")

learning_rate = args.learning_rate # Learning rate for the optimizer
per_device_train_batch_size = args.per_device_train_batch_size  # Batch size for training per device
per_device_eval_batch_size = args.per_device_eval_batch_size  # Batch size for evaluation per device
num_train_epochs = args.num_train_epochs  # Number of epochs for training
training_dataset_path = args.training_dataset_path
project_root = args.project_root
model_path = args.model_path
resume_from_checkpoint = args.resume_from_checkpoint
resume_checkpoint_path = args.resume_checkpoint_path
src_lng = args.src_lng
tgt_lng = args.tgt_lng
r = args.r
is_peft = args.is_peft
is_unsloth = args.is_unsloth
is_train_response_only = args.is_train_response_only

## Params need to be set
# learning_rate = 1e-5 # Learning rate for the optimizer
# per_device_train_batch_size = 10  # Batch size for training per device
# per_device_eval_batch_size = 10  # Batch size for evaluation per device
# num_train_epochs = 5  # Number of epochs for training
# training_dataset_path = "data/training_dataset/dataset_GPT_split.jsonl"
# project_root = "/home/snt/projects_lujun/mt_luxembourgish"
# model_path = "/home/snt/llm_models/Llama-3.2-1B-Instruct"
# resume_from_checkpoint = False
# resume_checkpoint_path = None
# src_lng = "English"
# tgt_lng = "Luxembourgish"
# r = 32
# is_peft = False
# is_unsloth = False
# is_train_response_only = True


model_name = model_path.split("/")[-1]
train_ratio = 0.005  # Number of samples to be used for training and evaluation
warmup_ratio = 0.5
logging_steps = 10
evaluation_strategy="steps"
save_strategy="epoch"
eval_steps=10
max_grad_norm = 0.3
fp16 = not is_bfloat16_supported()
MAX_LEN = 512
weight_decay = 0.01
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
train_seed = 3407

if is_peft:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",]
    lora_alpha = 8
    lora_dropout = 0
    random_state = 3407

current = time.time()
formatted_time = time.strftime("%m_%d_%H_%M", time.localtime(current))
if resume_from_checkpoint:
    output_dir = resume_checkpoint_path
else:
    if is_peft:
        input_file_name = training_dataset_path.split("/")[-1].split(".")[0]
        output_dir = f"logs/peft_{r}_{src_lng[:2]}_{tgt_lng[:2]}/fit_{formatted_time}_{train_ratio}_{input_file_name}"
    else:
        input_file_name = training_dataset_path.split("/")[-1].split(".")[0]
        output_dir = f"logs/full_{src_lng[:2]}_{tgt_lng[:2]}/fit_{formatted_time}_{train_ratio}_{input_file_name}"

if resume_from_checkpoint and resume_checkpoint_path is None:
    raise ValueError("Please provide a checkpoint path to resume training from")



# ========================== dataset preparation ==========================

train_dataset_path = os.path.abspath(os.path.join(project_root, training_dataset_path))
sys.path.append(project_root)

train_dataset_df = pd.read_json(train_dataset_path, lines=True)
pre_processed_dataset_df = pre_process(train_dataset_df)

if not isinstance(pre_processed_dataset_df, pd.DataFrame):
    raise TypeError("data_preprocess should return a pandas DataFrame.")

dataset = Dataset.from_pandas(pre_processed_dataset_df)

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

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = train_dataset.map(
    lambda sample: {
        "full_prompt": create_prompt(sample, src_lng=src_lng, tgt_lng=tgt_lng, mode="train", tokenizer=tokenizer)["full_prompt"]
    }
).select_columns(["full_prompt"])

val_dataset = val_dataset.map(
    lambda sample: {
        "full_prompt": create_prompt(sample, src_lng=src_lng, tgt_lng=tgt_lng, mode="train", tokenizer=tokenizer)["full_prompt"]
    }
).select_columns(["full_prompt"])

def tokenize_function(examples):
    return tokenizer(
        examples["full_prompt"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["full_prompt"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["full_prompt"])


from unsloth import FastLanguageModel
import torch
from peft import get_peft_model, LoraConfig


# Using Unsloth Acceleration 
if is_unsloth:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"unsloth/{model_name}", # or choose "unsloth/Llama-3.2-1B-Instruct"
        # model_name = model_path,
        max_seq_length = MAX_LEN,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    if is_peft:
        model = FastLanguageModel.get_peft_model(
            model,
            r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = target_modules,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = random_state,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And without LoftQ
        )

# Using transformer huggingface   
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.use_cache = False
    if is_peft:
        lora_config = LoraConfig(
            r=r, 
            target_modules=target_modules, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout, 
            bias="none", 
            random_state=random_state,
            use_rslora=False,
            loftq_config=None  # And without LoftQ
        )
        model = get_peft_model(model, lora_config)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print (model)
print(print_trainable_parameters(model))

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name}, Shape: {param.shape}, Trainable parameters: {param.numel()}")
    else:
        print(f"Layer: {name}, Shape: {param.shape}, Non-trainable parameters: {param.numel()}")


from transformers import logging as transformers_logging
from accelerate import Accelerator
from unsloth.chat_templates import train_on_responses_only
import warnings
warnings.simplefilter("ignore")
transformers_logging.set_verbosity_error()

def train_ddp_accelerate_sft():
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_ratio=warmup_ratio,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16= fp16,
        bf16 = is_bfloat16_supported(),
        max_grad_norm=max_grad_norm,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        remove_unused_columns=False,
        disable_tqdm=False,
        seed = train_seed,
        ddp_find_unused_parameters=False, # Avoids warnings
        dataloader_num_workers=4,  # Adjust number of workers based on hardware
        # load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer=tokenizer,
        train_dataset = tokenized_train_dataset,
        eval_dataset = tokenized_val_dataset,
        dataset_text_field = "full_prompt",
        max_seq_length = MAX_LEN,
        data_collator = data_collator,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
    )

    if is_train_response_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Finished training SFT.")
    return trainer_stats


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = None

def main():
    trainer_stats = train_ddp_accelerate_sft()
    return trainer_stats

if __name__ == "__main__":
    trainer_stats = main()


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# CUDA_VISIBLE_DEVICES=0 python notebook/training/SFT.py \
    # --per_device_train_batch_size 8 \
    # --per_device_eval_batch_size 8 \
    # --src_lng "English" \
    # --tgt_lng "Luxembourgish" \
    # --num_train_epochs 1 \
    # --learning_rate 1e-6 \
    # --project_root "/home/llama/Personal_Directories/srb/mt_luxembourgish" \
    # --training_dataset_path "data/training_dataset/dataset_llama_split.jsonl" \
    # --model_name "/home/llama/models/base_models/Llama-3.2-3B-Instruct" \
    # --resume_from_checkpoint True\
    # --resume_checkpoint_path "/home/snt/projects_lujun/mt_luxembourgish/logs/fit_1738867685.359803_0.001"
    # --r 256 \
    # --is_peft False \
    # --is_unsloth False \
    # --is_train_response_only True