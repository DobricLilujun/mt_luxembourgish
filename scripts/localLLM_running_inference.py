import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)
from langchain.prompts import PromptTemplate
import argparse


def get_latest_file(pattern: str) -> Optional[str]:
    """
    Get the latest file from the current directory that matches the given pattern.

    :param pattern: A string pattern (e.g., 'train.json', 'data_usda_crop').
    :return: The name of the most recently modified file matching the pattern, or None if no such file exists.
    """
    files = [f for f in os.listdir() if f.startswith(pattern)]  # List files in the current directory matching the pattern
    return max(files, key=os.path.getmtime) if files else None  # Return the most recent file based on modification time, or None if no matches

def load_checkpoint(latest_file: Optional[str], df: pd.DataFrame, text_column: str) -> int:
    """
    Load a checkpoint from a CSV file and return the starting index for translation.

    :param latest_file: Path to the most recent checkpoint file (CSV) or None if no checkpoint exists.
    :param df: DataFrame containing the texts (not used in this function but included for completeness).
    :param text_column: The column in the CSV containing the text to be translated.
    :return: The starting index for translation based on the latest checkpoint.
    """
    if latest_file:  # If a valid checkpoint file exists
        translated_df = pd.read_csv(latest_file)  # Load the checkpoint file
        translated_texts = translated_df[text_column].tolist()  # Get the translated texts from the specified column
        start_idx = len(translated_texts)  # Start index is the number of already translated texts
    else:
        start_idx = 0  # If no checkpoint exists, start from the beginning
    return start_idx

def validate_config(config: Dict[str, Optional[str]]) -> None:
    """
    Validates the configuration dictionary to ensure all necessary keys are present and valid.

    :param config: The configuration dictionary to validate.
    :raises ValueError: If any required key is missing or invalid.
    """
    required_keys = ["model_name", "text_column", "batch_size", "prefix", "device"]
    for key in required_keys:
        if key not in config or not config[key]:
            raise ValueError(f"Missing or invalid value for required config key: '{key}'")

    if not os.path.exists(config["model_name"]):  # Check if the model path exists
        raise ValueError(f"Model path does not exist: {config['model_name']}")

def generate_translation_prompt(text: str, language_1: str = "Luxembourgish", language_2: str = "English") -> str:
    """
    Generates a translation prompt using the specified languages.

    :param text: The text to be translated.
    :param language_1: The source language (default is Luxembourgish).
    :param language_2: The target language (default is English).
    :return: A formatted translation prompt.
    """
    # Template for the translation prompt
    prompt_template = """Please translate the following {language_1} text into {language_2}. Please answer me with only translated text!

    ---------------------------------- Text to be translated ----------------------------------

    {Text}

    ---------------------------------- End of Text ----------------------------------
    """
    
    # Create a PromptTemplate object with input variables and template
    translation_prompt = PromptTemplate(
        input_variables=["language_1", "language_2", "Text"],  # Corrected input variables to match template
        template=prompt_template
    )

    # Return the formatted prompt with the provided text and languages
    return translation_prompt.format(language_1=language_1, language_2=language_2, Text=text)




def initialize_pipeline(config: Dict[str, Optional[str]]) -> pipeline.Pipeline:
    """
    Initializes the LLM pipeline for text generation.

    :param config: Configuration dictionary containing model parameters.
    :return: An initialized pipeline object for text generation.
    """
    model_path = config["model_name"]
    load_in_4bit, load_in_8bit = config["current_load_in_4bit"], config["current_load_in_8bit"]

    # If loading quantization is enabled, initialize BitsAndBytesConfig for the model
    if config["if_loading_quantization"]:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit, 
            load_in_8bit=load_in_8bit, 
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        nf4_config = None  # No quantization if not enabled

    # Load the generation config from the pretrained model
    generation_config = GenerationConfig.from_pretrained(model_path)
    for key, value in config["model_config"].items():
        setattr(generation_config, key, value)

    # Initialize the text-generation pipeline
    text_pipeline = pipeline("text-generation", model=model_path, torch_dtype=torch.float32, device_map=config["device"])
    text_pipeline.generation_config = generation_config

    return text_pipeline

def get_available_gpus() -> List[int]:
    """
    Get the indices of available GPUs on the system using 'nvidia-smi'.

    :return: A list of GPU indices that are available, or an empty list if no GPUs are found.
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            return [int(gpu) for gpu in gpus]  # Return GPU indices as integers
        else:
            print(f"Error: {result.stderr}")
            return []  # Return empty list if there is an error with the 'nvidia-smi' command
    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure that NVIDIA drivers are installed.")
        return []  # Return empty list if 'nvidia-smi' is not found

def translate_batch_LLM(config: Dict[str, Optional[str]], df: pd.DataFrame) -> None:
    """
    Translate a batch of text from a DataFrame using the LLM pipeline.

    :param config: Configuration dictionary with model parameters.
    :param df: DataFrame containing the text to be translated in the specified column.
    :return: None (results are saved to a file).
    """
    # Initialize the translation pipeline
    translator = initialize_pipeline(config)
    df["prompts_inputs"] = df[config["text_column"]].apply(generate_translation_prompt)

    prefix = config["prefix"]
    latest_file = get_latest_file(prefix)
    print(f"Found latest file: {latest_file}")
    is_new_file_tag = bool(config.get("is_new_file", False))
    print(f"is_new_file tag: {is_new_file_tag}")
    
    # Determine output file based on the 'is_new_file' flag and the latest file found
    if latest_file and not bool(config.get("is_new_file", False)):
        output_file = latest_file
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{prefix}_{current_time}.csv"

    # Prepare the list of texts for translation
    texts = df[config["text_column"]].tolist()
    start_idx = (
        load_checkpoint(latest_file, df, config["text_column"]) 
        if (latest_file and not bool(config.get("is_new_file", False))) 
        else 0
    )

    print("Start From Index: ", start_idx)
    texts = df["prompts_inputs"].to_list()
    batch_size = config["batch_size"]
    
    available_gpus = get_available_gpus()
    print(f"Available GPUs: {available_gpus}")

    # Process texts in batches and perform translation
    for i in tqdm(range(start_idx, len(texts), batch_size), desc="Translating", unit="batch"):
        batch = texts[i:i + batch_size]
        translated_batch = translator(batch, pad_token_id=translator.tokenizer.eos_token_id, return_full_text=False)

        # Save the translated texts to the output file
        for j, text in enumerate(batch):
            updated_row = df.iloc[i + j].copy()
            updated_row["translated_text"] = translated_batch[j][0]['generated_text']
            updated_dataframe = pd.DataFrame([updated_row])
            
            # Write to the file, either appending or creating a new file
            if i == start_idx and j == 0 and bool(config.get("is_new_file", False)):
                updated_dataframe.to_csv(output_file, index=False, mode="w", header=True)
            else:
                updated_dataframe.to_csv(output_file, index=False, mode="a", header=False)

    print(f"Translation completed. Results saved to {output_file}")



def main() -> None:
    # Set the project root path
    project_root = "/home/users/luli/project/mt_luxembourgish"
    if project_root not in sys.path:  # Ensure the project root path is in the system path
        sys.path.append(project_root)

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

    # Create a config dictionary from parsed arguments
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