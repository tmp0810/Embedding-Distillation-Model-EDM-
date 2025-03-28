import torch
import torch.nn as nn
import torch.distributed as dist

class CrossEntropyLossForTextClassification(nn.Module):
    def __init__(self, args):
        super(CrossEntropyLossForTextClassification, self).__init__()
        self.label_smoothing = args.label_smoothing  # Tham số làm mịn nhãn (nếu có)
        self.num_classes = args.num_classes          # Số lượng lớp trong task phân loại

    def forward(self, logits, target, logging_output, batch_denom):

        # Tính mất mát cross-entropy
        loss, nll_loss = self.compute_cross_entropy_loss(logits, target)
        
        # Tính độ chính xác
        accuracy = self.compute_accuracy(logits, target)
        
        # Ghi lại thông tin logging
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "nll_loss": nll_loss,
                "accuracy": accuracy
            }
        )
        
        return loss / batch_denom, logging_output

    def compute_cross_entropy_loss(self, logits, target):

        # Tính log softmax trên chiều lớp
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        
        # Tính negative log likelihood loss
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        
        if self.label_smoothing > 0:
            # Tính mất mát mịn (smooth loss)
            smooth_loss = -lprobs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        
        return loss, nll_loss

    def compute_accuracy(self, logits, target):
        # Lấy chỉ số lớp có xác suất cao nhất
        pred = logits.argmax(dim=-1)
        
        # Tính số lượng mẫu dự đoán đúng
        correct = pred.eq(target).sum().float()
        return correct

    def record_logging_output(self, logging_output, batch_denom, content):
        for key, value in content.items():
            if key == "accuracy":
                # Chuẩn hóa accuracy dựa trên batch_denom
                record_value = value / batch_denom
            else:
                # Chuẩn hóa loss và nll_loss
                record_value = value / batch_denom
            
            # Đồng bộ giá trị giữa các GPU (nếu sử dụng distributed training)
            dist.all_reduce(record_value, dist.ReduceOp.SUM)
            record_value = record_value.item() / dist.get_world_size()
            
            # Lưu vào logging_output
            if key in logging_output:
                logging_output[key].append(record_value)
            else:
                logging_output[key] = [record_value]
        
        return logging_output