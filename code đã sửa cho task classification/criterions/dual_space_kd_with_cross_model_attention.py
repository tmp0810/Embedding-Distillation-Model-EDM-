import math
import torch
from .various_divergence import VariousDivergence


class DualSpaceKDWithCMA(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
        
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output
    
    def compute_dual_space_kd_loss_with_cma(
    self, outputs, teacher_outputs, input_data, output_data, distiller, log
):
        # Lấy nhãn classification
        target = output_data["label"]  # (batch_size,)
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]  # (batch_size,)

        # Lấy attention mask từ input_data
        student_input_mask = input_data["attention_mask"]  # (batch_size, sequence_length)
        teacher_input_mask = input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"]  # (batch_size, teacher_sequence_length)

        # Lấy hidden states của layer cuối cùng
        hiddens = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        teacher_hiddens = teacher_outputs.hidden_states[-1]  # (batch_size, teacher_sequence_length, hidden_size)

        # Trích xuất hidden state của token [CLS]
        stu_cls_hidden = hiddens[:, 0, :]  # (batch_size, hidden_size)
        tea_cls_hidden = teacher_hiddens[:, 0, :]  # (batch_size, hidden_size)

        # Tính alignment từ teacher sang student (t2s)
        stu_q_hiddens = distiller.projectors["query"](stu_cls_hidden).float()  # (batch_size, proj_dim)
        tea_k_hiddens = distiller.projectors["key"](teacher_hiddens).float()  # (batch_size, teacher_sequence_length, proj_dim)
        tea_v_hiddens = distiller.projectors["value"](teacher_hiddens).float()  # (batch_size, teacher_sequence_length, hidden_size)

        # Tính alignment giữa [CLS] của student và các token của teacher
        align = stu_q_hiddens.unsqueeze(1).matmul(tea_k_hiddens.transpose(-2, -1))  # (batch_size, 1, teacher_sequence_length)
        align = align / math.sqrt(distiller.projectors["query"].out_features)  # Chuẩn hóa theo chiều proj_dim
        align_mask = teacher_input_mask.float().unsqueeze(1)  # (batch_size, 1, teacher_sequence_length)
        align = align + (1.0 - align_mask) * (-100000)  # Mask các token padded của teacher

        t2s_weight = torch.softmax(align, -1)  # (batch_size, 1, teacher_sequence_length)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).squeeze(1)  # (batch_size, hidden_size)

        # Tính logits cho t2s
        # Giả sử sử dụng lm_head như code gốc, nhưng trong classification thông thường cần classification_head
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # (batch_size, num_classes hoặc vocab_size)

        # Tính cross-entropy loss cho t2s
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]  # Scalar
        t2s_acc = (t2s_logits.argmax(-1) == target).float().sum()  # Tổng số dự đoán đúng trong batch
        max_probs = t2s_logits.softmax(-1).max(-1)[0].sum()  # Tổng xác suất lớn nhất
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs

        # Tính thêm các loss khác nếu không chỉ train projectors
        if not self.args.only_save_projector:
            # Tính t2s_kd_loss giữa logits của student và t2s_logits
            t2s_kd_loss = self.dist_func(
                outputs.logits, t2s_logits.detach(), target, reduction="sum", use_tea_temp=True
            )

            # Tính alignment từ student sang teacher (s2t)
            tea_q_hiddens = distiller.projectors["query"](tea_cls_hidden).float()  # (batch_size, proj_dim)
            stu_k_hiddens = distiller.projectors["key"](hiddens).float()  # (batch_size, sequence_length, proj_dim)
            stu_v_hiddens = distiller.projectors["value"](hiddens).float()  # (batch_size, sequence_length, hidden_size)

            align_s2t = tea_q_hiddens.unsqueeze(1).matmul(stu_k_hiddens.transpose(-2, -1))  # (batch_size, 1, sequence_length)
            align_s2t = align_s2t / math.sqrt(distiller.projectors["query"].out_features)
            align_mask_s2t = student_input_mask.float().unsqueeze(1)  # (batch_size, 1, sequence_length)
            align_s2t = align_s2t + (1.0 - align_mask_s2t) * (-100000)

            s2t_weight = torch.softmax(align_s2t, -1)  # (batch_size, 1, sequence_length)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).squeeze(1)  # (batch_size, hidden_size)

            # Tính logits cho s2t
            s2t_logits = s2t_hiddens.matmul(
                distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
            )  # (batch_size, num_classes hoặc vocab_size)

            # Tính KL divergence giữa s2t_logits và teacher logits
            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, teacher_outputs.logits, teacher_target, reduction="sum"
            )
            s2t_acc = (s2t_logits.argmax(-1) == teacher_target).float().sum()

            # Tổng loss
            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc
        else:
            kd_loss = t2s_ce_loss

        log["kd_loss"] = kd_loss
        return kd_loss, log
    