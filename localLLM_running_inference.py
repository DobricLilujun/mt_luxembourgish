import argparse
import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from langchain.prompts import PromptTemplate

# Set project root path
project_root = "/home/users/luli/project/mt_luxembourgish"
if project_root not in sys.path:
    sys.path.append(project_root)

def get_latest_file(pattern):
    files = [f for f in os.listdir() if f.startswith(pattern)]
    return max(files, key=os.path.getmtime) if files else None

def load_checkpoint(latest_file, df, text_column):
    if latest_file:
        translated_df = pd.read_csv(latest_file)
        translated_texts = translated_df[text_column].tolist()
        start_idx = len(translated_texts)
    else:
        start_idx = 0
    return start_idx

def validate_config(config):
    """Validates the config dictionary."""
    required_keys = ["model_name", "text_column", "batch_size", "prefix", "device"]
    for key in required_keys:
        if key not in config or not config[key]:
            raise ValueError(f"Missing or invalid value for required config key: '{key}'")

    if not os.path.exists(config["model_name"]):
        raise ValueError(f"Model path does not exist: {config['model_name']}")


def generate_translation_prompt(text, language_1="Luxembourgish", language_2="English"):
    prompt_template = """Please translate the following {language_1} text into {language_2}. Please answer me with only translated text!

    ---------------------------------- Text to be translated ----------------------------------

    {Text}

    ---------------------------------- Text to be translated ----------------------------------

    """
    translation_prompt = PromptTemplate(
        input_variables=["language_1", "language_2", text],
        template=prompt_template
    )
    return translation_prompt.format(language_1=language_1, language_2=language_2, Text=text)


def initialize_pipeline(config):
    model_path = config["model_name"]
    load_in_4bit, load_in_8bit = config["current_load_in_4bit"], config["current_load_in_8bit"]

    if config["if_loading_quantization"]:
        nf4_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, bnb_4bit_compute_dtype=torch.float16)
    else:
        nf4_config = None

    generation_config = GenerationConfig.from_pretrained(model_path)
    for key, value in config["model_config"].items():
        setattr(generation_config, key, value)

    text_pipeline = pipeline("text-generation", model=model_path, torch_dtype=torch.float32, device_map=config["device"])
    text_pipeline.generation_config = generation_config
    return text_pipeline


def translate_batch_LLM(config, df):
    translator = initialize_pipeline(config)
    df["prompts_inputs"] = df[config["text_column"]].apply(generate_translation_prompt)

    prefix = config["prefix"]
    latest_file = get_latest_file(prefix)
    print (f"Found latest file: {latest_file}")
    is_new_file_tag = bool(config.get("is_new_file", False))
    print (f"is_new_file tag: {is_new_file_tag}")
    if latest_file and not bool(config.get("is_new_file", False)):
        output_file = latest_file
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{prefix}_{current_time}.csv"

    texts = df[config["text_column"]].tolist()
    start_idx = (
        load_checkpoint(latest_file, df, config["text_column"]) 
        if (latest_file and not bool(config.get("is_new_file", False))) 
        else 0
    )

    print("Start From Index: ", start_idx)
    texts = df["prompts_inputs"].to_list()
    batch_size = config["batch_size"]

    for i in tqdm(range(start_idx, len(texts), batch_size), desc="Translating", unit="batch"):
        batch = texts[i:i + batch_size]
        translated_batch = translator(batch, pad_token_id=translator.tokenizer.eos_token_id, return_full_text=False)

        for j, text in enumerate(batch):
            updated_row = df.iloc[i + j].copy()
            updated_row["translated_text"] = translated_batch[j][0]['generated_text']
            updated_dataframe = pd.DataFrame([updated_row])
            
            if i == start_idx and j == 0 and bool(config.get("is_new_file", False)):
                updated_dataframe.to_csv(output_file, index=False, mode="w", header=True)
            else:
                updated_dataframe.to_csv(output_file, index=False, mode="a", header=False)

    print(f"Translation completed. Results saved to {output_file}")


import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Batch translation using LLM pipeline")
    parser.add_argument("--model_name", required=True, help="Path to the model directory")
    parser.add_argument("--if_loading_quantization", type=bool, default=False, help="Enable quantization")
    parser.add_argument("--current_load_in_4bit", type=bool, default=False, help="Use 4-bit quantization")
    parser.add_argument("--current_load_in_8bit", type=bool, default=False, help="Use 8-bit quantization")
    parser.add_argument("--text_column", required=True, help="Name of the text column in the dataset")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for translation")
    parser.add_argument("--prefix", required=True, help="Prefix for output files")
    parser.add_argument("--device", default="auto", help="Device to use for inference (e.g., 'cpu', 'cuda')")
    parser.add_argument("--is_new_file", type=bool, default=False, help="Start a new output file")
    parser.add_argument("--input_file", required=True, help="Path to the input dataset CSV file")

    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "if_loading_quantization": args.if_loading_quantization,
        "current_load_in_4bit": args.current_load_in_4bit,
        "current_load_in_8bit": args.current_load_in_8bit,
        "model_config": {
            "temperature": 0.1,  # necessary
            "max_tokens": 512,  # necessary
            "top_p": 0.9,  # necessary
            "do_sample": True,  # necessary
            "max_new_tokens": 512,  # necessary
            "max_length": 512,  # necessary
        },
        "batch_size": args.batch_size,  # use this to accelerate the translation process
        "prefix": args.prefix,  # necessary
        "text_column": args.text_column,  # necessary
        "device": args.device,
        "is_new_file": args.is_new_file,
    }

    validate_config(config)  # Ensure this function is implemented
    df = pd.read_csv(args.input_file)
    translate_batch_LLM(config, df)  # Ensure this function is implemented

if __name__ == "__main__":
    main()

