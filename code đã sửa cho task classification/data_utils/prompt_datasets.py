import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.distributed import get_rank
from utils import log_rank
from tqdm import tqdm

class PromptDataset(Dataset):
    def __init__(self, args, split, tokenizer, data_path=None, num=-1):
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.data, self.label_map = self._load_and_process_data(data_path, num)
        log_rank(f"Num instances: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def _load_and_process_data(self, data_path, num):
        path = os.path.join(data_path, f"{self.split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file named {path}")

        # Load CSV data
        raw_data = pd.read_csv(path)
        if num != -1:
            raw_data = raw_data[:num]

        # Create label map: label string to token ID
        label_map = {
            label: self.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in raw_data['label'].unique()
        }
        data = []
        log_rank("Processing dataset for classification...")
        for _, row in tqdm(raw_data.iterrows(), total=len(raw_data), disable=(get_rank() != 0)):
            text = row['text']
            label = row['label']
            # Construct prompt
            prompt = text + " The answer is"
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            label_token_id = label_map[label]
            data.append({
                "prompt_ids": prompt_ids,
                "output_ids": [label_token_id]
            })
        return data, label_map

    def verbalizer(self):
        return {v: k for k, v in self.label_map.items()}

    def __getitem__(self, index):
        data = self.data[index]
        return index, data["prompt_ids"], data["output_ids"]

    def collate(self, samples):
        bs = len(samples)
        model_batch = {
            "input_ids": torch.full(
                (bs, self.max_prompt_length), self.pad_id, dtype=torch.long
            ),
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "labels": torch.full((bs, self.max_prompt_length + 1), -100, dtype=torch.long),
        }
        for i, (idx, prompt, rest) in enumerate(samples):
            prompt_len = min(len(prompt), self.max_prompt_length)
            prompt = prompt[-self.max_prompt_length:] if prompt_len == self.max_prompt_length else prompt
            # Left-pad prompt
            model_batch["input_ids"][i, -prompt_len:] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i, -prompt_len:] = 1
            # Set label at position after prompt
            no_model_batch["idx"][i] = idx
            no_model_batch["labels"][i, prompt_len] = rest[0]  # Label token ID
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)
        return model_batch, no_model_batch