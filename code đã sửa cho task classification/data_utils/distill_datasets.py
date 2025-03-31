import torch
import os
import csv
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm

from utils import log_rank
from typing import Dict, Optional, List
from transformers import AutoTokenizer


class DistillDataset(Dataset):
    def __init__(
        self, 
        args, 
        split: str,
        student_tokenizer: Dict[str, AutoTokenizer], 
        teacher_tokenizers: Optional[Dict[str, AutoTokenizer]] = {},
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.dataset, self.label_mapping = self._load_and_process_data()
        log_rank(f"Num of data instances: {len(self.dataset)}")
        log_rank(f"Number of classes: {len(self.label_mapping)}")

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        path = os.path.join(self.args.data_dir, f"{self.split}.csv")

        if os.path.exists(path):
            # Process CSV data
            with open(path, 'r', encoding='utf-8') as f:
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
            
            log_rank("Processing dataset for student model (and all teacher models)...")
            for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                # Encode the text
                student_text_ids = self.student_tokenizer.encode(
                    data["text"], add_special_tokens=True, truncation=True,
                    max_length=self.max_prompt_length
                )
                
                # Get label as numeric index
                label_idx = label_mapping[data["label"]]
                
                tokenized_data = {
                    "student_input_ids": student_text_ids,
                    "label": label_idx
                }
        
                # Process for teacher models
                for model_type in self.teacher_tokenizers:
                    if self.teacher_tokenizers[model_type] is None: 
                        continue
                    
                    teacher_text_ids = self.teacher_tokenizers[model_type].encode(
                        data["text"], add_special_tokens=True, truncation=True,
                        max_length=self.max_prompt_length
                    )
                    
                    tokenized_data[f"teacher_{model_type}_input_ids"] = teacher_text_ids

                dataset.append(tokenized_data)
            
            return dataset, label_mapping
        else:
            raise FileNotFoundError(f"No such file named {path}")
    
    def get_label_mapping(self):
        """Return the mapping from label strings to indices"""
        return self.label_mapping
    
    def _process_classification(
        self, i, samp, model_data, no_model_data, 
        teacher_model_data, teacher_no_model_data
    ):
        # Process student model data
        input_ids = np.array(samp["student_input_ids"])
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        
        model_data["input_ids"][i][:input_len] = torch.tensor(input_ids, dtype=torch.long)
        model_data["attention_mask"][i][:input_len] = 1.0
        
        # Store label
        no_model_data["labels"][i] = samp["label"]
        
        # Process teacher model data
        for model_type in self.teacher_tokenizers:
            t_input_ids = np.array(samp[f"teacher_{model_type}_input_ids"])
            t_input_ids = t_input_ids[:self.max_length]
            t_input_len = len(t_input_ids)
            
            teacher_model_data[model_type]["input_ids"][i][:t_input_len] = \
                torch.tensor(t_input_ids, dtype=torch.long)
            teacher_model_data[model_type]["attention_mask"][i][:t_input_len] = 1.0

            # Label is the same for all models
            teacher_no_model_data[model_type]["labels"][i] = samp["label"]

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)
                elif isinstance(data[k], dict):
                    for kk in data[k]:
                        data[k][kk] = data[k][kk].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                        * self.student_tokenizer.pad_token_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        # For classification we only need labels
        no_model_data = {
            "labels": torch.zeros(bs, dtype=torch.long)
        }
        
        teacher_model_data = {
            model_type: {
                "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                            * self.teacher_tokenizers[model_type].pad_token_id,
                "attention_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        teacher_no_model_data = {
            model_type: {
                "labels": torch.zeros(bs, dtype=torch.long)
            } for model_type in self.teacher_tokenizers
        }

        for i, samp in enumerate(samples):
            self._process_classification(
                i, samp, model_data, no_model_data, 
                teacher_model_data, teacher_no_model_data
            )

        # Combine all data for the batch
        for model_type in teacher_model_data:
            prefix = f"teacher_{model_type}_"
            for key in teacher_model_data[model_type]:
                model_data[f"{prefix}{key}"] = teacher_model_data[model_type][key]
                
            for key in teacher_no_model_data[model_type]:
                no_model_data[f"{prefix}{key}"] = teacher_no_model_data[model_type][key]
        
        return model_data, no_model_data, None  # Gen data is None for classification