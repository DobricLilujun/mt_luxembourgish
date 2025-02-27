import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import requests
import argparse
import json



def validate_config(config):
    """Validate the configuration dictionary."""
    required_keys = ["model_name", "server_url"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration: {key}")

    if not config["server_url"].startswith("http"):
        raise ValueError("`server_url` must be a valid URL.")

    if not os.path.exists(config["model_name"]):
        raise ValueError(f"Model path does not exist: {config['model_name']}")


def generate_text_with_vllm(config, prompt):
    """Generate text using the specified language model."""
    headers = {"Content-Type": "application/json"}
    if "gemma" in config["model_name"]:
        payload = {
            "model": config["model_name"],
            "messages": [
                ##{"role": "assistant", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
    else:
        payload = {
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
    payload.update(config["options"])
    response = requests.post(config["server_url"], headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def get_latest_file(prefix):
    """Get the most recent file based on a filename prefix."""
    files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith(".jsonl")]
    return max(files, key=os.path.getmtime) if files else None


def load_checkpoint(latest_file, text_column):
    """Load checkpoint from the latest generated file."""
    if latest_file:
        translated_df = pd.read_json(latest_file,lines = True)
        translated_texts = translated_df[text_column].tolist()
        return len(translated_texts)
    return 0


def main(args):
    # ===================== Configurable Parameters =====================
    # User-configurable parameters to update
    MODEL_NAME = args.model_name  # Model name
    MODEL_BASE_PATH = args.model_path_base  # Base path for the model
    INPUT_FILE_PATH = args.input_file_path
    # DATA_PATH_TEMPLATE = f"{args.input_path_folder}/{{}}.csv"  # Data path template

    SEED_LIST = [
        47 + i for i in range(args.rep)
    ]  # Seed values for reproducibility

    SERVER_URL = args.server_url
    TEXT_COLUMN = "input"  # Column name for input text
    IS_NEW_FILE = args.is_new_file  # Whether to create a new output file
    PREFIX = (
        f"vllm_exp_dataset_csv_{MODEL_NAME}_"  # Prefix for generated output file names
    )
    OUTPUT_OPTIONS = {  # Configuration options for model generation
        "temperature": 0.3,
        "max_tokens": 512,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "seed": 47,  # Default
        "n": 1,
    }

    exp_df = pd.read_json(INPUT_FILE_PATH, lines=True)

    cartesian_df_final = pd.concat(
        [exp_df.assign(seed=seed) for seed in SEED_LIST], ignore_index=True
    )

    # Config dictionary
    config = {
        "model_name": MODEL_BASE_PATH + MODEL_NAME,
        "server_url": SERVER_URL,
        "prefix": PREFIX,
        "text_column": TEXT_COLUMN,
        "is_new_file": IS_NEW_FILE,
        "options": OUTPUT_OPTIONS,
    }

    # Validate the configuration
    validate_config(config)

    # Get the latest output file or start fresh
    latest_file = get_latest_file(PREFIX)
    if config.get("is_new_file", False):
        output_file = f"{PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        start_idx= 0
    else:
        if latest_file:
            output_file = latest_file
            start_idx = (
                load_checkpoint(
                    latest_file,
                    config["text_column"],
                ))
        else:
            output_file = f"{PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            start_idx = 0
    print("Start From Index: ", start_idx)


# python vllm_running_inference.py --model_name Llama-3.2-3B-Instruct-awq --model_path_base /home/llama/models/base_models/ --input_path_folder /home/llama/Personal_Directories/srb/temperature_eval/data/Intermediate/ --rep 3 --server_url http://0.0.0.0:4362/v1/chat/completions

    for i, row in tqdm(
        cartesian_df_final.iterrows(),
        total=cartesian_df_final.shape[0],
        desc="Processing Rows",
    ):
        if i < start_idx:
            continue
        updated_row = row.copy()
        config["options"]["seed"] = updated_row["seed"]
        generated_text = generate_text_with_vllm(config, updated_row["input"])
        updated_row["generate_response"] = generated_text
        updated_dataframe = pd.DataFrame([updated_row])

        if config.get("is_new_file", False):
            if i == start_idx:
                mode = "w"
            else:
                mode = "a"
        else:
            mode = "a"
            
        updated_dataframe.to_json(
            output_file,
            orient="records",
            lines=True,
            mode=mode,
        )

    print(f"Tasks completed. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Translation Script")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the model",
        default="Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--model_path_base",
        type=str,
        required=True,
        help="Path to the model",
        default="/home/snt/llm_models/",
    )
    parser.add_argument(
        "--input_file_path",
        type=str,
        required=True,
        help="Path to the model",
        default="/home/snt/projects_lujun/temperature_eval/data/Intermediate/test.jsonl",
    )
    parser.add_argument(
        "--rep",
        type=int,
        required=True,
        default=3,
        help="Repetition times to create more responses",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="vllm server URL",
        default="http://0.0.0.0:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--is_new_file",
        action="store_true",
        help="Start a new output file",
        default=False,
    )
    args = parser.parse_args()
    main(args)


# python scripts/vllm_running_inference_prompt.py --model_name "Llama-3.2-1B-Instruct" --model_path_base "/home/snt/llm_models/" --input_file_path "/home/snt/projects_lujun/mt_luxembourgish/data/flores-200-translation.jsonl" --rep 3 --server_url "http://0.0.0.0:8000/v1/chat/completions"
