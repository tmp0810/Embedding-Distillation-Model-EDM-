# -*- coding: utf-8 -*-
"""Mistral_QASC.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g6lLVHgUnkudn2HA24ZkPntMVDT3Vywb
"""

from huggingface_hub import login

login(token="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")

!pip install -q transformers accelerate trl bitsandbytes datasets evaluate peft scikit-learn
!pip install llm2vec

from llm2vec import LLM2Vec

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
)
model = model.merge_and_unload()  # This can take several minutes on cpu

# Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
)

# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=128)

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# Apply LoRA for fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
model = get_peft_model(model, lora_config)

# Load the QASC dataset
dataset = load_dataset("allenai/qasc")
train_dataset = dataset["train"]
test_dataset = dataset["validation"]

# Preprocess the dataset
def preprocess_batch(batch):
    questions = batch["question"]
    choices_list = batch["choices"]  # List of choice lists
    answer_keys = batch["answerKey"]

    # Debug output
    print("Batch keys:", batch.keys())
    print("First choices:", choices_list[0])

    # Process choices
    all_choices = []
    choice_offsets = [0]
    targets = []

    for i in range(len(questions)):
        if isinstance(choices_list[i], list):  # Standard QASC format
            choices = [choice["text"] for choice in choices_list[i]]
        elif isinstance(choices_list[i], dict):  # Alternative format
            choices = choices_list[i]["text"]
        else:
            raise ValueError(f"Unexpected choices format: {type(choices_list[i])}")

        all_choices.extend(choices)
        choice_offsets.append(choice_offsets[-1] + len(choices))
        answer_idx = ord(answer_keys[i]) - ord("A")
        targets.append(answer_idx)

    return questions, all_choices, targets, choice_offsets

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop with batching
batch_size = 16
model.train()
for epoch in range(1):
    total_loss = 0
    for i in tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch+1}"):
        batch = train_dataset[i:i + batch_size]

        # Process batch
        questions, all_choices, targets, choice_offsets = preprocess_batch(batch)

        # Tokenize all texts in the batch
        texts = questions + all_choices
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Mean pooling
        embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

        # Split embeddings
        question_embs = embeddings[:len(questions)]
        choice_embs = embeddings[len(questions):]

        # Compute scores for each question
        batch_scores = []
        for j in range(len(questions)):
            start_idx = choice_offsets[j]
            end_idx = choice_offsets[j + 1]
            q_emb = question_embs[j].unsqueeze(0)
            c_embs = choice_embs[start_idx:end_idx]
            scores = torch.nn.functional.cosine_similarity(q_emb, c_embs, dim=-1)
            batch_scores.append(scores)

        # Stack scores and targets
        scores = torch.stack(batch_scores)
        targets = torch.tensor(targets, dtype=torch.long).to(model.device)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(scores, targets)
        total_loss += loss.item() * len(questions)  # Scale by batch size

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_dataset)}")

# Evaluation loop (single-example for simplicity)
def preprocess_example(example):
    question = example["question"]
    if isinstance(example["choices"], list):
        choices = [choice["text"] for choice in example["choices"]]
    elif isinstance(example["choices"], dict):
        choices = example["choices"]["text"]
    else:
        raise ValueError(f"Unexpected choices format: {type(example['choices'])}")
    answer_key = example["answerKey"]
    answer_idx = ord(answer_key) - ord("A")
    return question, choices, answer_idx

model.eval()
ranks = []              # For MRR
correct_at_1 = 0        # For Precision@1
correct_at_3 = 0        # For Precision@3
correct_at_5 = 0        # For Precision@5
total = 0               # Total examples

with torch.no_grad():
    for example in tqdm(test_dataset, desc="Evaluating"):
        question, choices, answer_idx = preprocess_example(example)
        texts = [question] + choices

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        question_emb = embeddings[0]
        choice_embs = embeddings[1:]

        scores = torch.nn.functional.cosine_similarity(question_emb.unsqueeze(0), choice_embs, dim=-1)

        _, indices = torch.sort(scores, descending=True)

        # Compute rank of correct answer (1-based)
        rank = (indices == answer_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

        # Precision@k: Check if correct answer is in top-k
        top_1 = indices[:1]   # Top 1 prediction
        top_3 = indices[:3]   # Top 3 predictions
        top_5 = indices[:5]   # Top 5 predictions

        if answer_idx in top_1:
            correct_at_1 += 1
        if answer_idx in top_3:
            correct_at_3 += 1
        if answer_idx in top_5:
            correct_at_5 += 1

        total += 1

# Calculate metrics
mrr = np.mean([1.0 / rank for rank in ranks])
precision_at_1 = correct_at_1 / total
precision_at_3 = correct_at_3 / total
precision_at_5 = correct_at_5 / total

# Print results
print("")
print(f"MRR: {mrr:.4f}")
print(f"Precision@1: {precision_at_1:.4f}")
print(f"Precision@3: {precision_at_3:.4f}")
print(f"Precision@5: {precision_at_5:.4f}")