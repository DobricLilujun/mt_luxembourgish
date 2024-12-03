from huggingface_hub import login
import os
import time
import pandas as pd
import torch.cuda
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import random
import functools
import csv
import pandas as pd
import numpy as np
import torch.nn.functional as F
import evaluate
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)



import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, load_dataset

# Create DataLoaders
batch_size = 32  
MAX_LEN = 512
# Load validation and training datasets
val_dataset = load_dataset("facebook/flores", "all")
val_dataset = val_dataset["devtest"].select_columns(["sentence_eng_Latn", "sentence_ltz_Latn"])

train_dataset = load_from_disk("/home/lujun_li/projects/mt_luxembourgish/data/fake_targets/NC_LUX.arrow").select_columns(["subsentence", "translated_text"]).rename_columns({
    "subsentence": "sentence_ltz_Latn",  # Renaming 'subsentence' to 'sentence_eng_Latn'
    "translated_text": "sentence_eng_Latn"  # Renaming 'translated_text' to 'sentence_ltz_Latn'
}).select([i for i in range(1000)])

# Convert datasets to dictionaries
dataset = DatasetDict({ 'train': train_dataset, 'val': val_dataset})


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

current = time.time()
writer = SummaryWriter(log_dir=f"logs/fit_{current}/")

learning_rate = 1e-4
per_device_train_batch_size = 50
per_device_eval_batch_size = 10
num_train_epochs = 100
weight_decay = 0.01


model_name = "/home/lujun_li/projects/base_models/Llama-3.2-1B-Instruct"
# Quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

# Lora
lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS',
)


model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quantization_config,)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1




# Create LLAMA tokenized dataset which will save time
def llama_preprocessing_function(examples):
    # Tokenize both English and Luxembourgish sentences
    tokenized_input = tokenizer(examples['sentence_ltz_Latn'], truncation=True, max_length=MAX_LEN, return_tensors="pt", padding='max_length')  # max length set to 512 for accelerating training
    tokenized_target = tokenizer(examples['sentence_eng_Latn'], truncation=True, max_length=MAX_LEN, return_tensors="pt", padding='max_length')
    
    # Return the tokenized sentences
    return {
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'labels': tokenized_target['input_ids']
    }

tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True)
tokenized_datasets.set_format("torch")




class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Extract the input and target sentences
        input_ids = inputs.get("input_ids")  # English sentences as input
        target_ids = inputs.get("labels")  # Luxembourgish sentences as target labels

        # Perform a forward pass through the model
        outputs = model(input_ids=input_ids, attention_mask=inputs.get('attention_mask'))
        # Get the logits from the model outputs
        logits = outputs.logits

        # Compute the loss using CrossEntropyLoss, ignore padding tokens
        loss_fct = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=model.config.pad_token_id)

        # Return the loss and outputs if return_outputs is True, otherwise just the loss
        return (loss_fct, outputs) if return_outputs else loss_fct


    def log(self, logs):
        super().log(logs)
        # Log each key-value pair in logs using the writer
        for key, value in logs.items():
            self.writer.add_scalar(key, value, self.state.global_step)





collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

# define training args
training_args = TrainingArguments(
    output_dir = f"logs/fit_{current}",
    learning_rate = learning_rate,
    per_device_train_batch_size = per_device_train_batch_size,
    per_device_eval_batch_size = per_device_eval_batch_size,
    num_train_epochs = num_train_epochs,
    weight_decay = weight_decay,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    report_to="tensorboard",
    ddp_find_unused_parameters=False,
)

# Define custom trainer
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
)
train_result = trainer.train()
writer.close()