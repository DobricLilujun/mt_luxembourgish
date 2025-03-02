import pandas as pd
import re

## Remove HTML possbile tag inputs
## Remove hallucination data from the dataset because of the Llama models or other decoder-only models
def pre_process(df):

    # Compute input length
    df["input_length"] = df["input"].apply(len)
    
    # Define HTML tag pattern
    html_pattern = r"</?[a-z][\s\S]*?>"

    # Identify rows containing HTML tags
    df["is_contain_html"] = df["input"].str.contains(html_pattern, case=False, na=False, regex=True)
    
    # Identify hallucinated data
    df["is_hallucinate"] = df["translated_text"].str.len() > 2.0 * df["input"].str.len()
    
    # Print dataset size before cleaning
    count_before = len(df)
    print(f"Length of inputs before: {count_before}")
    
    # Remove unwanted rows
    df = df[~df["is_hallucinate"] & ~df["is_contain_html"]]
    
    # Print dataset size after cleaning
    count_after = len(df)
    print(f"Length of inputs after: {count_after}")
    print(f"Removed rows: {count_before - count_after}")
    
    return df

def create_prompt(sample, src_lng, tgt_lng, is_prefix =True, is_suffix = True, eos_rep = 3, mode="train", tokenizer=None):

    if tokenizer is None or tokenizer.eos_token is None:
        raise ValueError("A tokenizer with a defined EOS token is required.")

    system_message = f"You are a helpful AI assistant for translation."
    response_prefix = "Here is the translation: " if is_prefix else ""
    response_suffix = f"\nEnd of translation." if is_suffix else ""

    input_text = sample[src_lng.capitalize()].strip()  # Extract the input text.
    response = ( sample[tgt_lng.capitalize()].strip() if tgt_lng.capitalize() in sample else "")  # Extract the target text.
    question = f"Translate the following English input text into Luxembourgish. Do not include any additional information or unrelated content.\n\n{input_text}"
    # Get the EOS token from the tokenizer.
    eos_token = tokenizer.eos_token
    if mode == "train":
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response_prefix + response + response_suffix + eos_token* eos_rep}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
            ]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return { "full_prompt": full_prompt }


def create_prompt_gemma(sample, src_lng, tgt_lng, is_prefix =True, is_suffix = True, eos_rep = 3, mode="train", tokenizer=None):

    if tokenizer is None or tokenizer.eos_token is None:
        raise ValueError("A tokenizer with a defined EOS token is required.")
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    input_text = sample[src_lng.capitalize()].strip()  # Extract the input text.
    response = ( sample[tgt_lng.capitalize()].strip() if tgt_lng.capitalize() in sample else "")  # Extract the target text.
    instruction = "Translate the following English input text into Luxembourgish. Do not include any additional information or unrelated content."
    # Get the EOS token from the tokenizer.
    eos_token = tokenizer.eos_token
    if mode == "train":
        full_prompt = alpaca_prompt.format(instruction, input_text, response) + eos_token
    else:
        full_prompt = alpaca_prompt.format(instruction, input_text, "") + eos_token

    return { "full_prompt": full_prompt }

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
