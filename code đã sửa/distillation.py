import time
import os

from sklearn.metrics import precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import shutil
import json
from tqdm import tqdm
import math
from transformers import (
    AutoTokenizer,
    GenerationConfig
)
from transformers.integrations import HfDeepSpeedConfig
from arguments import get_args
from distiller import Distiller
from data_utils.distill_datasets import DistillDataset
from utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather
)
from criterions import build_criterion
from rouge_metric import compute_metrics

torch.set_num_threads(4)

# Chuẩn bị data
def prepare_dataset(args, distiller):
    data = {}
    if args.do_train:
        data["train"] = DistillDataset(
            args, "train", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of train data: {}".format(len(data["train"])))
        
        data["dev"] = DistillDataset(
            args, "dev", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of dev data: {}".format(len(data["dev"])))

        if os.path.exists(os.path.join(args.data_dir, "test.jsonl")):
            data["test"] = DistillDataset(
                args, "test", distiller.student_tokenizer,
                distiller.teacher_tokenizers
            )
            log_rank("Num of test data: {}".format(len(data["test"])))

    elif args.do_eval:
        data["test"] = DistillDataset(
            args, "test", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of test data: {}".format(len(data["test"])))
    else:
        raise ValueError("Do train and do eval must set one")
        
    return data

def finetune(
    args, 
    tokenizer: AutoTokenizer, 
    model: deepspeed.DeepSpeedEngine, 
    optimizer: AdamW, 
    lr_scheduler, 
    dataset, 
    device, 
):
    log_rank("Start Fine-tuning")
    start_time = time.time()

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        criterion = build_criterion(args)

    sampler = DistributedSampler(
        dataset["train"], 
        shuffle=True, 
        drop_last=True, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    train_dataloader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )
    
    step = 0
    model_list = []

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        log_rank("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        end_epoch = False
        epoch_step = 0
        epoch_loss, epoch_nll_loss, epoch_kd_loss = 0.0, 0.0, 0.0
        train_iter = iter(train_dataloader)
        st_time = time.time()
        loss, logging_output = model(
            criterion, batch, logging_output, loss_denom)
        model.backward(loss)
        model.step()

        torch.cuda.synchronize()
        elapsed_time = time.time() - st_time
        logging_output["micro_step_time"].append(elapsed_time)
        step += 1
        logging_output["global_step"] += 1
        logging_output["step_time"].append(time.time() - global_st_time)
        epoch_step += 1

        # Đánh giá và lưu mô hình
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            if (epoch + 1) % args.eval_interval == 0:
                log_rank("Evaluating before saving model...")
                # Conduct evaluation before saving model
                eval_loss, eval_results = evaluate(
                    args, 
                    tokenizer, 
                    model.module.student_model, 
                    dataset["dev"], 
                    "dev", 
                    device
                )
                if "test" in dataset:
                    _, _ = evaluate(
                        args, 
                        tokenizer, 
                        model.module.student_model, 
                        dataset["test"], 
                        "test", 
                        device,
                        repeat_times=1
                    )

                # Lưu checkpoint
                if args.eval_gen:
                    ckpt_name = "epoch{}_step{}_loss{:.4f}_rougel{:.4f}".format(
                        epoch + 1, 
                        logging_output["global_step"], 
                        eval_loss, 
                        eval_results["rougeL"]
                    )
                else:
                    ckpt_name = "epoch{}_step{}_loss{:.4f}".format(
                        epoch + 1, 
                        logging_output["global_step"], 
                        eval_loss
                    )
                save_dir_path = os.path.join(args.save_dir, ckpt_name)
                
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    if not args.only_save_projector:
                        log_rank("Saving tokenizer...")
                        tokenizer.save_pretrained(save_dir_path)
                        log_rank("Saving model...")
                        model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                    if hasattr(model.module, "projectors"):
                        log_rank("Saving projector...")
                        torch.save(
                            model.module.projectors.state_dict(), 
                            os.path.join(save_dir_path, "projector.pt")
                        )
                    
                    # QUẢN LÝ Checkpoint
                    # only keep best N checkpoints
                    if args.eval_gen:
                        model_list.append({
                            "path": save_dir_path, 
                            "score": eval_results["rougeL"]
                        })
                        model_list = sorted(model_list, key=lambda x: x["score"])
                    else:
                        model_list.append({
                            "path": save_dir_path, 
                            "score": eval_loss
                        })
                        model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)
                        
                    if len(model_list) > args.keep_best_n_checkpoints:
                        removed_model = model_list.pop(0)
                        shutil.rmtree(removed_model["path"])

                    log_rank(f"Model has been saved to {save_dir_path}")
                dist.barrier()
            else:
                ckpt_name = "epoch{}_step{}".format(
                    epoch + 1, 
                    logging_output["global_step"], 
                )
                save_dir_path = os.path.join(args.save_dir, ckpt_name)
                
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    if not args.only_save_projector:
                        log_rank("Saving tokenizer...")
                        tokenizer.save_pretrained(save_dir_path)
                        log_rank("Saving model...")
                        model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                    if hasattr(model.module, "projectors"):
                        log_rank("Saving projector...")
                        torch.save(
                            model.module.projectors.state_dict(), 
                            os.path.join(save_dir_path, "projector.pt")
                        )
                    # only keep best N checkpoints
                    model_list.append({
                        "path": save_dir_path, 
                        "score": logging_output["global_step"]
                    })
                    model_list = sorted(model_list, key=lambda x: x["score"])
                        
                    if len(model_list) > args.keep_best_n_checkpoints:
                        removed_model = model_list.pop(0)
                        shutil.rmtree(removed_model["path"])

                    log_rank(f"Model has been saved to {save_dir_path}")
                dist.barrier()

    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

@torch.no_grad

def evaluate(args, tokenizer, model, dataset, split, device):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss(reduction="sum")

    print(f"Evaluating on {split} set with {dp_world_size} GPU(s)")

    sampler = DistributedSampler(
        dataset,
        shuffle=False,
        drop_last=False,
        rank=dp_rank,
        num_replicas=dp_world_size
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    model.eval()
    eval_info = {
        "loss": 0.0,
        "sample_num": 0,
        "correct_samples": 0
    }

    all_preds = []
    all_labels = []

    for input_batch, output_batch in dataloader:
        dataset.move_to_device([input_batch, output_batch], device)

        logits = model(
            input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            position_ids=input_batch.get("position_ids", None)
        )

        labels = output_batch["labels"]

        loss = loss_func(logits, labels)

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()

        all_preds.append(preds)
        all_labels.append(labels)

        sample_num = labels.size(0)
        
        eval_info["loss"] += loss.item()
        eval_info["sample_num"] += sample_num
        eval_info["correct_samples"] += correct

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_preds_gathered = [torch.zeros_like(all_preds) for _ in range(dp_world_size)]
    all_labels_gathered = [torch.zeros_like(all_labels) for _ in range(dp_world_size)]
    dist.all_gather(all_preds_gathered, all_preds, group=dp_group)
    dist.all_gather(all_labels_gathered, all_labels, group=dp_group)

    all_preds = torch.cat(all_preds_gathered, dim=0)
    all_labels = torch.cat(all_labels_gathered, dim=0)

    if dp_rank == 0:
        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        precision = precision_score(all_labels_np, all_preds_np, average='macro')
        recall = recall_score(all_labels_np, all_preds_np, average='macro')

        eval_info["precision"] = round(precision, 6)
        eval_info["recall"] = round(recall, 6)

    eval_info["loss"] /= eval_info["sample_num"]
    eval_info["accuracy"] = eval_info["correct_samples"] / eval_info["sample_num"]

    for key in eval_info:
        if isinstance(eval_info[key], float):
            eval_info[key] = round(eval_info[key], 6)

    print(f"{split} | {eval_info}")

    model.train()
    return eval_info["loss"], eval_info["accuracy"], eval_info.get("precision", 0.0), eval_info.get("recall", 0.0)


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    # save arguments
    if dist.get_rank() == 0:
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]
    if "bf16" in ds_config:
        args.fp32 = not ds_config["bf16"]["enabled"]
    log_rank(args)
    args.deepspeed_config = None
    
    # prepare for deepspeed ZeRO-3
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    
    log_rank("Initializing a distiller for knowledge distillation...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(
            len(dataset["train"]) / 
            (args.batch_size * dp_world_size * args.gradient_accumulation_steps)
        )
        log_rank("Train iters per epoch = {}".format(args.train_iters_per_epoch))

        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        log_rank("Total_iters = {}".format(args.total_iters))
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    optimizer = get_optimizer(args, distiller.student_model)
    optimizer = distiller.add_optimizer_param_group(optimizer)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model, optimizer, lr_scheduler, dataset, device)
   
    if args.do_eval:
        evaluate(
            args, 
            distiller.student_tokenizer, 
            model, 
            dataset["test"], 
            "test", 
            0, 
            device
        )
        
    
if __name__ == "__main__":
    main()
