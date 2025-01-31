from typing import Dict, Optional, List, Union
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os
import sys

# Add the project root directory to the system path
project_root = "/home/users/luli/project/mt_luxembourgish"
if project_root not in sys.path:
    sys.path.append(project_root)

def get_latest_file(pattern: str) -> Optional[str]:
    """
    Finds the most recently modified file with a specific prefix.

    Input:
        - pattern (str): The prefix for file names to search.

    Output:
        - str: The name of the most recent file matching the prefix.
        - None: If no files match the given prefix.
    """
    files = [f for f in os.listdir() if f.startswith(pattern)]
    return max(files, key=os.path.getmtime) if files else None

def load_checkpoint(latest_file: Optional[str], df: pd.DataFrame, text_column: str) -> int:
    """
    Determines the starting index for translation based on the existing output file.

    Input:
        - latest_file (Optional[str]): Path to the most recent output file. Can be None if no previous file exists.
        - df (pd.DataFrame): The input DataFrame containing text data.
        - text_column (str): The column name in the DataFrame that contains the text to be translated.

    Output:
        - int: The starting index for translation (equal to the number of already translated rows in the previous file).
    """
    if latest_file:
        translated_df = pd.read_csv(latest_file)
        translated_texts = translated_df[text_column].tolist()
        start_idx = len(translated_texts)
    else:
        start_idx = 0
    return start_idx

def translate_batch(config: Dict[str, Union[str, int, bool]], df: pd.DataFrame) -> None:
    """
    Translates text in batches using a pre-trained translation model and saves the output to a CSV file.

    Input:
        - config (Dict[str, Union[str, int, bool]]): Configuration dictionary with the following keys:
            - model_name (str): Path to the model directory.
            - src_lang (str): Source language code.
            - tgt_lang (str): Target language code.
            - device (str): Device for translation ('cpu' or 'cuda').
            - max_length (int): Maximum length of translated text.
            - batch_size (int): Size of each batch for translation.
            - text_column (str): Column in the DataFrame to translate.
            - prefix (str): Prefix for output files.
            - is_new_file (bool): Flag to determine whether to create a new output file.
        - df (pd.DataFrame): Input data containing the text to translate.

    Output:
        - None: Translated results are saved to a CSV file.
    """
    # Load model and tokenizer
    model_path = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Initialize the translation pipeline
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=config["src_lang"],
        tgt_lang=config["tgt_lang"],
        max_length=config["max_length"],
        device=config["device"]
    )

    # Determine the output file name
    prefix = config["prefix"]
    latest_file = get_latest_file(prefix)
    if latest_file and not bool(config.get("is_new_file", False)):
        output_file = latest_file
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{prefix}_{current_time}.csv"

    # Check the starting index for translation
    texts = df[config["text_column"]].tolist()
    start_idx = (
        load_checkpoint(latest_file, df, config["text_column"])
        if (latest_file and not bool(config.get("is_new_file", False)))
        else 0
    )

    print("Start From Index: ", start_idx)

    # Translate text in batches and save the results
    batch_size = config["batch_size"]
    for i in tqdm(range(start_idx, len(texts), batch_size), desc="Translating", unit="batch"):
        batch = texts[i:i + batch_size]
        translated_batch = translator(batch)

        for j, text in enumerate(batch):
            updated_row = df.iloc[i + j].copy()
            updated_row["translated_text"] = translated_batch[j]["translation_text"]
            updated_dataframe = pd.DataFrame([updated_row])
            updated_dataframe.to_json(output_file, lines=True, mode="a", index=False)
            
    print(f"Translation completed. Results saved to {output_file}")

def main() -> None:
    """
    Main function to parse command-line arguments and execute the translation process.

    Input:
        - None (arguments are taken from the command line).

    Output:
        - None: Translates text and saves the output to a file.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch Translation Script")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--src_lang", type=str, required=True, help="Source language code")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for translation (e.g., 'cpu' or 'cuda:0')")
    parser.add_argument("--max_length", type=int, default=360, help="Maximum length for translation")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for translation")
    parser.add_argument("--text_column", type=str, default="subsentence", help="Column name in the dataframe to translate")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for output files")
    parser.add_argument("--is_new_file", type=bool, default=False, help="Flag to determine if a new file should be created")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file containing text to translate")

    args = parser.parse_args()

    # Create a configuration dictionary
    config = {
        "model_name": args.model_name,
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "device": args.device,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "text_column": args.text_column,
        "prefix": args.prefix,
        "is_new_file": args.is_new_file
    }

    # Load the input dataset
    if args.input_file.endswith(".jsonl"):
        dataset_df = pd.read_json(args.input_file, lines=True)
    else:
        dataset_df = pd.read_csv(args.input_file)

    # Run the translation process
    translate_batch(config, dataset_df)

if __name__ == "__main__":
    main()


# python nllb_running_inference.py --model_name "" --src_lang "lb" --tgt_lang "en" --device "cuda:0" --max_length 1024 --batch_size 6 --text_column "subsentence" --prefix "nllb_en" --input_file "data/processed/RTL2024_subsentences.jsonl"