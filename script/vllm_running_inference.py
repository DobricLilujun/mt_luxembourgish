import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from langchain.prompts import PromptTemplate
import os
import requests
import json
import sys
import argparse


def validate_config(config):
    """Validate the configuration dictionary."""
    required_keys = ["model_name", "server_url", "batch_size", "prefix", "text_column"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration: {key}")
    
    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    
    if not config["server_url"].startswith("http"):
        raise ValueError("`server_url` must be a valid URL.")
    
    if not os.path.exists(config["model_name"]):
        raise ValueError(f"Model path does not exist: {config['model_name']}")


def generate_translation_prompt(text, language_1="Luxembourgish", language_2="English"):
    """Generate a translation prompt."""
    prompt_template = """Please translate the following {language_1} text into {language_2}. Please answer me with only translated text!

    ---------------------------------- Text to be translated ----------------------------------

    {Text}

    ---------------------------------- Text to be translated ----------------------------------

    """
    translation_prompt = PromptTemplate(
        input_variables=["language_1", "language_2", "Text"],
        template=prompt_template
    )
    return translation_prompt.format(language_1=language_1, language_2=language_2, Text=text)


def generate_text_with_vllm(config, prompt):
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model_name"],
        "prompt": prompt,
    }
    payload.update(config["options"])

    response = requests.post(config["server_url"], headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("generated_text", "")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def get_latest_file(prefix):
    files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith(".csv")]
    return max(files, key=os.path.getmtime) if files else None


def load_checkpoint(latest_file, df, text_column):
    if latest_file:
        translated_df = pd.read_csv(latest_file)
        translated_texts = translated_df[text_column].tolist()
        return len(translated_texts)
    return 0


def translate_batch_vllm(config, df):
    df["prompts_inputs"] = df[config["text_column"]].apply(generate_translation_prompt)

    prefix = config["prefix"]
    latest_file = get_latest_file(prefix)
    output_file = (
        latest_file
        if latest_file and not config.get("is_new_file", False)
        else f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    start_idx = (
        load_checkpoint(latest_file, df, config["text_column"])
        if latest_file and not config.get("is_new_file", False)
        else 0
    )

    print("Start From Index: ", start_idx)
    texts = df["prompts_inputs"].to_list()

    for i in tqdm(range(start_idx, len(texts), config["batch_size"]), desc="Translating", unit="batch"):
        batch = texts[i:i + config["batch_size"]]
        translated_batch = [generate_text_with_vllm(config, text) for text in batch]

        for j, text in enumerate(batch):
            updated_row = df.iloc[i + j].copy()
            updated_row["translated_text"] = translated_batch[j]
            updated_dataframe = pd.DataFrame([updated_row])
            
            write_mode = "w" if i == start_idx and j == 0 and bool(config.get("is_new_file", False)) else "a"
            updated_dataframe.to_csv(output_file, index=False, mode=write_mode, header=(write_mode == "w"))

    print(f"Translation completed. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Translation Script")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model")
    parser.add_argument("--server_url", type=str, required=True, help="vllm server URL")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for translation")
    parser.add_argument("--prefix", type=str, required=True, help="Output file prefix")
    parser.add_argument("--text_column", type=str, required=True, help="Column name with text to translate")
    parser.add_argument("--is_new_file", action="store_true", help="Start a new output file")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling value")

    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "server_url": args.server_url,
        "batch_size": args.batch_size,
        "prefix": args.prefix,
        "text_column": args.text_column,
        "is_new_file": args.is_new_file,
        "options": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
        }
    }

    try:
        validate_config(config)
        dataset_df = pd.read_csv(args.input_file)
        translate_batch_vllm(config=config, df=dataset_df)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
