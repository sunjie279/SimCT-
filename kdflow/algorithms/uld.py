import torch
import torch.nn.functional as F

from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)


@register_algorithm("uld")
class ULD:
    """Universal Logit Distillation (ULD) for cross-tokenizer distillation.

    Reference: "Towards Cross-Tokenizer Distillation: the Universal Logit Distillation
    Loss for LLMs" (Boizard et al., TMLR 2025)

    Uses the Wasserstein-1 distance (closed-form) between sorted teacher and student
    probability distributions at each aligned token position.
    """

    def __init__(
        self,
        strategy,
        student_model,
        teacher_lm_head,
        student_tokenizer,
        teacher_tokenizer,
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        self.uld_lambda = getattr(self.args.kd, "uld_lambda", 1.5)
        self.uld_temperature = getattr(self.args.kd, "uld_temperature", 1.0)
        self.uld_top_k = getattr(self.args.kd, "uld_top_k", 1024)

        logger.info(
            f"[ULD] Initialized with lambda={self.uld_lambda}, "
            f"temperature={self.uld_temperature}, "
            f"top_k={self.uld_top_k}"
        )

    def _align_sequences(self, tea_seq, stu_seq):
        """Greedy sequence alignment between teacher and student token sequences."""
        i, j = 0, 0
        t2s_align, s2t_align = [], []
        history_tea_seq, history_stu_seq = "", ""

        tea_eos = self.teacher_tokenizer.eos_token
        stu_eos = self.student_tokenizer.eos_token
        EOS = "<|eos|>"

        tea_seq = [EOS if token == tea_eos else token.replace('\u2581', '').replace('\u0120', '') for token in tea_seq]
        stu_seq = [EOS if token == stu_eos else token.replace('\u2581', '').replace('\u0120', '') for token in stu_seq]

        if tea_seq == stu_seq:
            indices = list(range(len(tea_seq)))
            return indices, indices

        while i < len(tea_seq) and j < len(stu_seq):
            if history_tea_seq == history_stu_seq and tea_seq[i] == stu_seq[j]:
                common_text = tea_seq[i]
                history_tea_seq += common_text
                history_stu_seq += common_text
                t2s_align.append(i)
                s2t_align.append(j)
                i += 1
                j += 1
            elif len(history_tea_seq) > len(history_stu_seq):
                history_stu_seq += stu_seq[j]
                j += 1
            elif len(history_tea_seq) < len(history_stu_seq):
                history_tea_seq += tea_seq[i]
                i += 1
            else:
                history_tea_seq += tea_seq[i]
                history_stu_seq += stu_seq[j]
                i += 1
                j += 1

        return t2s_align, s2t_align

    def _compute_wasserstein_1(self, student_logits, teacher_logits):
        """Compute Wasserstein-1 distance with closed-form solution.

        Under uniform cost assumption, W1 = sum_i |p_sorted_i - q_sorted_i|
        where both distributions are padded to the same size and sorted in
        decreasing order.

        Memory optimization: uses top-k approximation to avoid sorting/padding
        the full vocabulary (e.g. 256K). Only the top-k largest probabilities
        are kept; the remaining probability mass is aggregated into a single
        "residual" bin. This is highly accurate because LLM logit distributions
        are extremely concentrated (top-1024 typically covers >99.9% mass).
        """
        temperature = self.uld_temperature
        top_k = self.uld_top_k

        # Compute softmax over full vocab (in-place, no extra copy)
        student_probs = F.softmax(student_logits / temperature, dim=-1, dtype=torch.float32)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)

        vocab_s = student_probs.shape[-1]
        vocab_t = teacher_probs.shape[-1]

        if top_k > 0 and top_k < min(vocab_s, vocab_t):
            # Top-k approximation: keep top-k probs, aggregate rest into residual
            stu_topk_vals, _ = student_probs.topk(top_k, dim=-1, sorted=True)  # [N, K] descending
            tea_topk_vals, _ = teacher_probs.topk(top_k, dim=-1, sorted=True)  # [N, K] descending

            # Residual mass = 1 - sum(top_k_probs), spread as a single bin
            stu_residual = (1.0 - stu_topk_vals.sum(dim=-1, keepdim=True)).clamp(min=0)
            tea_residual = (1.0 - tea_topk_vals.sum(dim=-1, keepdim=True)).clamp(min=0)

            # Append residual as the (K+1)-th bin
            student_sorted = torch.cat([stu_topk_vals, stu_residual], dim=-1)  # [N, K+1]
            teacher_sorted = torch.cat([tea_topk_vals, tea_residual], dim=-1)  # [N, K+1]

            # Free full-vocab tensors immediately
            del student_probs, teacher_probs, stu_topk_vals, tea_topk_vals
        else:
            # Full-vocab fallback (original behavior)
            max_vocab = max(vocab_s, vocab_t)

            if vocab_s < max_vocab:
                padding = torch.zeros(
                    student_probs.shape[0], max_vocab - vocab_s,
                    device=student_probs.device, dtype=student_probs.dtype
                )
                student_probs = torch.cat([student_probs, padding], dim=-1)

            if vocab_t < max_vocab:
                padding = torch.zeros(
                    teacher_probs.shape[0], max_vocab - vocab_t,
                    device=teacher_probs.device, dtype=teacher_probs.dtype
                )
                teacher_probs = torch.cat([teacher_probs, padding], dim=-1)

            student_sorted, _ = student_probs.sort(dim=-1, descending=True)
            teacher_sorted, _ = teacher_probs.sort(dim=-1, descending=True)

        w1_per_position = torch.abs(student_sorted - teacher_sorted).sum(dim=-1)

        return w1_per_position

    def training_step(self, micro_batch):
        """Perform one training step with ULD loss."""
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_attn_mask = micro_batch["tea_attn_mask"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for ULD"

        mm_kwargs = {k[3:]: v for k, v in micro_batch.items() if k.startswith("mm_")}

        # Forward pass through student model
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"]

        # Compute teacher logits from hidden states
        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits = self.teacher_lm_head(teacher_hiddens)

        # Align sequences
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)[teacher_loss_mask]

        teacher_aligned_idx, student_aligned_idx = self._align_sequences(
            self.teacher_tokenizer.convert_ids_to_tokens(teacher_label_ids.cpu().tolist()),
            self.student_tokenizer.convert_ids_to_tokens(student_label_ids.cpu().tolist()),
        )

        if len(teacher_aligned_idx) == 0 or len(student_aligned_idx) == 0:
            kd_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            align_ratio = torch.tensor(0.0, device=student_logits.device)
            return {"loss": kd_loss, "kd_loss": kd_loss, "align_ratio": align_ratio}

        # Extract aligned logits
        student_logits_flat = student_logits[student_loss_mask]
        aligned_student_logits = student_logits_flat[student_aligned_idx]
        aligned_teacher_logits = teacher_logits[teacher_aligned_idx]

        # Compute Wasserstein-1 distance
        w1_distances = self._compute_wasserstein_1(aligned_student_logits, aligned_teacher_logits)

        align_ratio = torch.tensor(
            len(student_aligned_idx) / max(len(student_label_ids), 1),
            device=student_logits.device,
        )

        kd_loss = self.uld_lambda * w1_distances.sum() / avg_token_num

        loss_info = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "align_ratio": align_ratio,
        }

        if self.args.kd.kd_ratio < 1:
            ce_loss = compute_cross_entropy(
                student_logits_flat, student_label_ids, reduction="sum"
            ) / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
