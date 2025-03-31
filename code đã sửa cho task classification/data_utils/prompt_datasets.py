import torch
import os
import csv
from torch.utils.data import Dataset

from torch.distributed import get_rank
from utils import log_rank
from tqdm import tqdm


class PromptDataset(Dataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length

        # Load CSV data
        self.data, self.labels, self.label_mapping = self.load_data_csv(data_path, num)
        self.num_classes = len(self.label_mapping)
        
        self.num = len(self.data)
        if num != -1 and num < self.num:
            self.num = num
            self.data = self.data[:self.num]
            self.labels = self.labels[:self.num]
            
        log_rank(f"Num instances: {len(self.data)}")
        log_rank(f"Number of classes: {self.num_classes}")
        log_rank(f"Class mapping: {self.label_mapping}")
            
    def __len__(self):
        return self.num

    def load_data_csv(self, data_path, data_num):
        data_path = os.path.join(data_path, f"{self.split}.csv")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No such file: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            
            # Identify text and label columns (assuming first column is text, second is label)
            text_col_idx = 0  # Default to first column
            label_col_idx = 1  # Default to second column
            
            if hasattr(self.args, 'text_column') and self.args.text_column in header:
                text_col_idx = header.index(self.args.text_column)
            if hasattr(self.args, 'label_column') and self.args.label_column in header:
                label_col_idx = header.index(self.args.label_column)
                
            raw_data = []
            for row in csv_reader:
                if len(row) > max(text_col_idx, label_col_idx):
                    raw_data.append({
                        "text": row[text_col_idx],
                        "label": row[label_col_idx]
                    })
        
        # Create label mapping
        unique_labels = sorted(set(item["label"] for item in raw_data))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        processed_data = []
        processed_labels = []
        
        log_rank("Loading and processing data")
        for item in tqdm(raw_data[:data_num] if data_num != -1 else raw_data, 
                         desc="Loading Data", disable=(get_rank() != 0)):
            text = item["text"]
            label = label_mapping[item["label"]]
            
            # Encode text
            text_ids = self.tokenizer.encode(
                text, 
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length
            )
            
            processed_data.append(text_ids)
            processed_labels.append(label)
            
        log_rank("Load End")
        return processed_data, processed_labels, label_mapping

    def get_num_classes(self):
        return self.num_classes
    
    def get_label_mapping(self):
        return self.label_mapping

    def __getitem__(self, index: int):
        text_ids = self.data[index]
        label = self.labels[index]
        
        # Truncate if needed
        text_ids = text_ids[:self.max_length]
        
        return index, text_ids, label
    
    def collate(self, samples):
        bs = len(samples)
        
        max_text_length = max([len(samp[1]) for samp in samples])
        max_text_length = min(max_text_length, self.max_length)
        
        model_batch = {
            "input_ids": torch.ones(bs, max_text_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_text_length, dtype=torch.long),
        }
        
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "labels": torch.zeros(bs, dtype=torch.long)
        }
        
        for i, (idx, text_ids, label) in enumerate(samples):
            text_len = min(len(text_ids), max_text_length)
            model_batch["input_ids"][i][:text_len] = torch.tensor(text_ids[:text_len], dtype=torch.long)
            model_batch["attention_mask"][i][:text_len] = 1
            
            no_model_batch["idx"][i] = idx
            no_model_batch["labels"][i] = label
        
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch