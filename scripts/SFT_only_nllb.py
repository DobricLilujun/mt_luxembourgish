# ====================================================IMPORT=================================================================
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.update(
    {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "1",
    }
)

import time
import torch
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from accelerate import Accelerator
from trl import SFTConfig, SFTTrainer
from sacrebleu.metrics import BLEU

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import EarlyStoppingCallback
from accelerate import Accelerator

# ====================================================SETTINGS=================================================================
current = time.time()
output_dir = f"logs/fit_{current}"
resume_from_checkpoint = False
# output_dir = "/home/llama/Personal_Directories/srb/mt_luxembourgish/logs/fit_1733825724.5355668"

learning_rate = 2e-5
per_device_train_batch_size = 30
per_device_eval_batch_size = 30
num_train_epochs = 3
weight_decay = 0.01
MAX_LEN = 512
# sample_number = 10000
lora_r = 32  # 16, 32
lora_alpha = 16  # 8, 16
lora_dropout = 0.1  # 0.05, 0.1
model_name = (
    "/home/llama/Personal_Directories/srb/binary_classfication/Llama-3.2-3B-Instruct"
)
val_dataset_path = (
    "/home/llama/Personal_Directories/srb/mt_luxembourgish/data/flores_devtest_arrow"
)
train_dataset_path = (
    "/home/llama/Personal_Directories/srb/mt_luxembourgish/data/nllb_df_complete_subsentences_filtered.arrow"
)

# bitsandbytes parameters
use_4bit = True  # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"  # Quantization type (fp4 or nf4)
use_nested_quant = (
    False  # Activate nested quantization for 4-bit base models (double quantization)
)

# ====================================================LOAD MODEL=================================================================

compute_dtype = getattr(
    torch, bnb_4bit_compute_dtype
)  # Load tokenizer and model with QLoRA configuration

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_use_double_quant=use_nested_quant,
    bnb_4bit_compute_dtype=compute_dtype,
)

# Setting up the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ====================================================DATA=================================================================
# Load validation and training datasets

# val_dataset = load_from_disk("/home/llama/Personal_Directories/srb/mt_luxembourgish/data/flores_devtest_arrow").select([i for i in range(sample_number)])
val_dataset = (
    load_from_disk(
        # "/home/llama/Personal_Directories/srb/mt_luxembourgish/data/flores_devtest_arrow"
        val_dataset_path
    )
    .rename_columns(
        {
            "sentence_ltz_Latn": "Luxembourgish",  # Renaming 'subsentence' to 'sentence_eng_Latn'
            "sentence_eng_Latn": "English",  # Renaming 'translated_text' to 'sentence_ltz_Latn'
        }
    )
)
train_dataset = (
    load_from_disk(
        # "/home/llama/Personal_Directories/srb/mt_luxembourgish/data/NC_LUX.arrow"
        train_dataset_path
    )
    .select_columns(["subsentence", "translated_text"])
    .rename_columns(
        {
            "subsentence": "Luxembourgish",  # Renaming 'subsentence' to 'sentence_eng_Latn'
            "translated_text": "English",  # Renaming 'translated_text' to 'sentence_ltz_Latn'
        }
    )
)


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

# Convert datasets to dictionaries
dataset = DatasetDict({"train": train_dataset, "val": val_dataset})


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
    # Load LoRA configuration
    # Setting up the model
    # device_map = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map=device_map,
        quantization_config=bnb_config,
        # pretraining_tp=1,
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        task_type="CAUSAL_LM",
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("Your GPU supports bfloat16: accelerate training with bf16=True")

    # Unused TrainingArguments parameters
    # gradient_checkpointing = True  # Enable gradient checkpointing 节省内存

    # Hyper Parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,  # Model predictions and checkpoints will be stored
        num_train_epochs=num_train_epochs,  # Number of epochs
        # max_steps=-1,                     # Number of training steps (overrides num_train_epochs)
        per_device_train_batch_size=per_device_train_batch_size,  # Batch size per GPU for training
        per_device_eval_batch_size=per_device_eval_batch_size,  # Batch size per GPU for evaluation
        # gradient_checkpointing = True,       # Enable gradient checkpointing
        warmup_ratio=0.03,  # Ratio of steps for a linear warmup (from 0 to learning rate)
        logging_steps=100,  # Log every X updates steps
        # save_steps=1000,                     # Save checkpoint every X updates steps
        save_strategy="epoch",
        evaluation_strategy="epoch",
        eval_steps=500,
        # gradient_accumulation_steps=1,  # Number of update steps to accumulate the gradients for
        # optim="paged_adamw_32bit",  # Optimizer to use
        learning_rate=learning_rate,  # Initial learning rate (AdamW optimizer)
        weight_decay=0.001,  # Weight decay to apply to all layers except bias/LayerNorm weights
        fp16=False,  # Use mixed precision (bfloat16)
        bf16=True,  # Enable fp16/bf16 training (set bf16 to True with an A100)
        max_grad_norm=0.3,  # Maximum gradient normal (gradient clipping)
        group_by_length=True,  # Group sequences into batches with same length (Saves memory and speeds up training considerably)
        lr_scheduler_type="cosine",  # Learning rate schedule
        report_to="tensorboard",
        # load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
        # greater_is_better=True
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    print ("------------------"*5)
    print(print_trainable_parameters(model))
    print ("------------------"*5)

    # # Define the early stopping callback
    # early_stopping = EarlyStoppingCallback(
    #     early_stopping_patience=3,
    #     early_stopping_threshold=0.005,
    # )

    # # Modify your evaluation function to ensure it returns a dictionary with 'accuracy'
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return {"accuracy": (predictions == labels).mean()}  # Ensure this is the name expected by your Trainer setup

    # # Quick test with a reduced number of steps
    # training_arguments.num_train_epochs = 1  # Reduce for a quick test
    # training_arguments.eval_steps = 50       # Evaluate more frequently to test early stopping
    # training_arguments.save_steps = 50       # Save more frequently to ensure saving logic works

    # Setup the trainer with the prepared datasets

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        dataset_text_field="prompt_response",  # This now matches the transformed dataset
        max_seq_length=4 * MAX_LEN,
        processing_class=tokenizer,
        args=training_arguments,
        packing=False,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        # compute_metrics=compute_metrics,  # Ensure your trainer class uses this function correctly
        # callbacks=[early_stopping]

    )

    trainer.processing_class = tokenizer

    # Train the model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


from accelerate import Accelerator


def main():
    accelerator = Accelerator()
    train_ddp_accelerate()


if __name__ == "__main__":
    main()
