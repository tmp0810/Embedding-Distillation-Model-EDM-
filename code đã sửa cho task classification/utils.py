import numpy as np
import os
import time
import torch.distributed as dist
import random
import logging
import torch
import torch.nn as nn
from datetime import timedelta
import deepspeed
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_constant_schedule_with_warmup, 
    get_polynomial_decay_schedule_with_warmup,
    AutoModelForSequenceClassification,
    AutoModel
)


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s]  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Logging
def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


def log_rank(content, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        logging.info(content)


# Distributed
def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t


# Initialize
def set_random_seed(seed, mp=False):
    """Set random seed for reproducability."""
    seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def init_distributed(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    log_rank(f"Using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=300))


def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    log_rank(f"Using world size: {args.world_size}")
    
    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()

    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=30))


def initialize(args):
    # init bmt
    if args.deepspeed:
        init_distributed_ds(args)
    else:
        init_distributed(args)

    if args.model_parallel:
        raise NotImplementedError

    set_random_seed(args.seed, args.model_parallel)
    # init save folder
    if args.save_dir != None:
        os.makedirs(args.save_dir, exist_ok=True)


# Load and save model
'''
        Sửa cả đoạn này cho đúng BERT của mình
'''
def get_model(args, device):
    #Lấy cả num_label
    config = AutoConfig.from_pretrained(args.model_path)
    
    st_time = time.time()
    if args.model_parallel:
        raise NotImplementedError
    else:
        
        config.is_model_parallel = False
        dtype = torch.float32 if args.fp32 else torch.float16

        try:
            model = AutoModel.from_pretrained(
                args.model_path, 
                config=config, 
                device_map={"": device}, 
                torch_dtype=dtype,
                
            )
        except:
            model = AutoModel.from_pretrained(
                args.model_path, 
                config=config, 
                device_map={"": device}, 
                torch_dtype=dtype,
                
            )
            model = model.half()
        
        
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    ed_time = time.time()
    
    log_rank(f"Model load time: {ed_time - st_time}s")
    
    return model


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        
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
        model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

    model.eval()
    
    return model

                
def get_optimizer_params(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters


def get_optimizer_params_peft(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    log_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters
        )
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min
        )
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5
        )
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

