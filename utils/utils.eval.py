from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

import pandas as pds
from tqdm import tqdm
import sacrebleu
from datasets import Dataset
from datasets import load_from_disk
from sacrebleu.metrics import CHRF
from datasets import load_dataset
from datetime import datetime
import json
from unsloth import FastLanguageModel



MAX_LEN = 512
model_path = "/home/snt/llm_models/Llama-3.2-3B-Instruct"
val_dataset_path = "data/training_dataset/dataset_val_300.jsonl"
flore_dataset_path = "data/fake_targets/flores_devtest_arrow"
current_time = datetime.now()
formatted_time = current_time.strftime('%m_%d_%H_%M')
eval_output_path = val_dataset_path.split("/")[-1].replace(".jsonl", f"_{formatted_time}_eval_from_Llama3-3B.jsonl")
sample_num = None  # Number of samples to evaluate otherwise set to None


src_lng = "English"
src_lng_abr = "sentence_eng_Latn"

# src_lng = "Luxembourgish"
# src_lng_abr = "sentence_ltz_Latn"

tgt_lng = "Luxembourgish"
tgt_lng_abr = "sentence_ltz_Latn"

# tgt_lng = "English"
# tgt_lng_abr = "sentence_eng_Latn"

device="cuda:0"


# def load_checkpoint(model_path, is_unsloth, is_peft, device):

#     if is_unsloth:
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = model_path,
#             max_seq_length = MAX_LEN
#         )
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         tokenizer.pad_token = tokenizer.eos_token
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map=device,
#         )
    





