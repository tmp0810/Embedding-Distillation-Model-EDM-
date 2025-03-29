import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

def main(local_rank, world_size):
    print('==REMOVEME local_rank', local_rank)
    # [1] Khởi tạo tiến trình phân tán
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    # dist.init_process_group(backend="nccl", )  # Thường dùng NCCL cho GPU
    # local_rank = int(os.environ["LOCAL_RANK"])  # Biến môi trường tự động gán bởi torchrun
    # torch.cuda.set_device(local_rank)          # Chọn GPU theo local_rank
    device = torch.device("cuda", local_rank)
    print('==REMOVEME device', device)

    # [2] Tải & xử lý dữ liệu
    dataset = load_dataset('yelp_review_full')
    if dist.get_rank() == 0:
        print("Dataset loaded—ready to roll!")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets['test']

    # Sử dụng DistributedSampler để dữ liệu được chia đều cho mỗi GPU
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, sampler=test_sampler, shuffle=False)

    # [3] Khởi tạo model, optimizer
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    model.to(device)

    # Bọc mô hình trong DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Hàm tính metric
    def compute_metrics(labels, preds):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # [4] Vòng lặp huấn luyện
    num_epochs = 3
    for epoch in range(num_epochs):
        # Cập nhật seed/sampler để shuffle nhất quán giữa các tiến trình
        train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        
        # Chỉ rank=0 in thông báo
        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}: Đang huấn luyện...")

        # Thanh tiến độ chỉ xuất hiện ở rank=0 (disable=True với tiến trình khác)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=(dist.get_rank() != 0)):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Tính loss trung bình & in ra ở rank=0
        avg_train_loss = total_loss / len(train_dataloader)
        if dist.get_rank() == 0:
            print(f"Epoch {epoch + 1} hoàn thành! Training Loss: {avg_train_loss:.4f}")

        # [5] Đánh giá
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        metrics = compute_metrics(all_labels, all_preds)
        if dist.get_rank() == 0:
            print("Kết quả đánh giá:")
            for key, value in metrics.items():
                print(f"  {key.capitalize()}: {value:.4f}")

    # [6] Kết thúc, lưu model chỉ ở rank=0
    if dist.get_rank() == 0:
        print("\nKết quả đánh giá cuối cùng:")
        for key, value in metrics.items():
            print(f"  {key.capitalize()}: {value:.4f}")

        save_directory = "./kaggle/working/yelp-review-classi"
        # Model gốc nằm trong model.module (khi đã bọc DDP)
        model.module.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('==REMOVEME world_size', world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size)