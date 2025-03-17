import json
import os
import sys
import requests
from typing import Optional
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def select_sharegpt_promots(file_path, nums, target_prompt_len, tokenizer):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data.get("conversations", data.get("conversation", []))) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]
    filtered_dataset = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == nums:
            break
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)
        if prompt_len < target_prompt_len:
            # Prune too short sequences.
            continue
        elif prompt_len > target_prompt_len:
            # Pad too long sequences.
            prompt_token_ids = prompt_token_ids[:target_prompt_len]
        else:
            # Keep the sequence as is.
            pass
        target_prompt = tokenizer.decode(prompt_token_ids)
        filtered_dataset.append(target_prompt)
    return filtered_dataset

def get_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if __name__ == "__main__":
    model_path = 'meta-llama/Llama-2-7b-chat-hf'
    tokenizer = get_tokenizer(model_path)
    
    file_path = os.path.join(os.path.dirname(__file__), "ShareGPT_V3_unfiltered_cleaned_split.json")
    if not os.path.isfile(file_path):
        file_path = download_and_cache_file(SHAREGPT_URL, file_path)

    nums = 10
    target_prompt_len = 100
    print(select_sharegpt_promots(file_path, nums, target_prompt_len, tokenizer))