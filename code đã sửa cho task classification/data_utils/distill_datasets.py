import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer
import json

class DistillDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        student_tokenizer: AutoTokenizer,  # Simplified to a single tokenizer object
        teacher_tokenizers: Optional[Dict[str, AutoTokenizer]] = None,
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers or {}
        self.max_length = args.max_length
        self.dataset = self._load_and_process_data()
        log_rank(f"Num of data instances: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def _load_and_process_data(self):
        dataset = []
        path = os.path.join(self.args.data_dir, f"{self.split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file named {path}")

        # Load CSV data
        raw_data = pd.read_csv(path)
        # Create label map from unique labels
        self.label_map = {label: i for i, label in enumerate(raw_data['label'].unique())}

        log_rank("Processing dataset for student model (and teacher models if applicable)...")
        for _, row in tqdm(raw_data.iterrows(), total=len(raw_data), disable=(dist.get_rank() != 0)):
            text = row['text']
            label = self.label_map[row['label']]
            # Tokenize input text
            tokenized = self.student_tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            data_instance = {
                "input_ids": tokenized['input_ids'][0],  
                "attention_mask": tokenized['attention_mask'][0],
                "label": label
            }
            # Optionally include teacher probabilities
            for model_type in self.teacher_tokenizers:
                prob_col = f"teacher_{model_type}_probs"
                if prob_col in row and row[prob_col]:
                    data_instance[prob_col] = json.loads(row[prob_col])
            dataset.append(data_instance)
        return dataset

    def collate(self, samples):
        bs = len(samples)
        # Prepare model inputs
        model_data = {
            "input_ids": torch.stack([samp['input_ids'] for samp in samples]),
            "attention_mask": torch.stack([samp['attention_mask'] for samp in samples]),
        }
        # Prepare labels
        no_model_data = {
            "labels": torch.tensor([samp['label'] for samp in samples], dtype=torch.long)
        }
        # Add teacher probabilities if present
        for model_type in self.teacher_tokenizers:
            prob_key = f"teacher_{model_type}_probs"
            if prob_key in samples[0]:
                no_model_data[prob_key] = torch.stack([
                    torch.tensor(samp[prob_key], dtype=torch.float) for samp in samples
                ])
        return model_data, no_model_data

    def move_to_device(self, datazip, device):
        model_data, no_model_data = datazip
        for data in (model_data, no_model_data):
            for k in data:
                data[k] = data[k].to(device)
        return model_data, no_model_data