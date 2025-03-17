import json
import os
import sys
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
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
    nums = 10
    target_prompt_len = 100
    select_sharegpt_promots(file_path, nums, target_prompt_len, tokenizer)