import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
import pandas as pd
from datetime import datetime
import os
import sys


# python nllb_running_inference.py --model_name /mnt/lscratch/users/luli/model/nllb-200-3.3B  --src_lang ltz_Latn  --tgt_lang eng_Latn  --device cuda --max_length 360  --batch_size 6 --text_column subsentence --is_new_file True --prefix translation_nllb_ --input_file NC_lux_subsentences_test.csv
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

def translate_batch(config, df):
    # Initialize model and tokenizer
    model_path = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=config["src_lang"],
        tgt_lang=config["tgt_lang"],
        max_length=config["max_length"],
        device=config["device"]
    )

    # Determine output file name 
    prefix = config["prefix"]
    latest_file = get_latest_file(prefix)
    if latest_file and not bool(config.get("is_new_file", False)):
        output_file = latest_file
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{prefix}_{current_time}.csv"

    # Load data and check starting index
    texts = df[config["text_column"]].tolist()
    start_idx = (
        load_checkpoint(latest_file, df, config["text_column"]) 
        if (latest_file and not bool(config.get("is_new_file", False))) 
        else 0
    )

    print("Start From Index: ", start_idx)
    texts = df[config["text_column"]].to_list()
    
    # Batch translation and saving
    batch_size = config["batch_size"]
    for i in tqdm(range(start_idx, len(texts), batch_size), desc="Translating", unit="batch"):
        batch = texts[i:i + batch_size]
        translated_batch = translator(batch)
        
        for j, text in enumerate(batch):
            updated_row = df.iloc[i + j].copy()
            updated_row["translated_text"] = translated_batch[j]["translation_text"]
            updated_dataframe = pd.DataFrame([updated_row])
            
            mode = "w" if i == start_idx and j == 0 and start_idx == 0 else "a"
            header = mode == "w"
            updated_dataframe.to_csv(output_file, index=False, mode=mode, header=header)

    print(f"Translation completed. Results saved to {output_file}")

def main():
    # Setup command-line argument parser
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
    
    # Parse arguments
    args = parser.parse_args()

    # Config dictionary from command line arguments
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

    # Load the dataset
    dataset_df = pd.read_csv(args.input_file)

    # Run the translation
    translate_batch(config, dataset_df)

if __name__ == "__main__":
    main()
