import torch
import torch.nn.functional as F

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


@register_algorithm("span_ctkd_1to1")
class SpanCTKD1to1(object):
    """Ablation variant of SpanCrossTokenizerKD: uses the more accurate
    ``tokenizer.decode``-based alignment (instead of convert_ids_to_tokens +
    strip), but **only keeps 1:1 aligned positions** and discards all span
    segments.

    Purpose: verify whether the span segments' loss signal is the source of
    noise that degrades performance compared to simple_ctkd.
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
        self.student_overlap_token_ids, self.teacher_overlap_token_ids = self._find_overlap_tokens()
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
        self._debug_count = 0

    # ------------------------------------------------------------------
    # Overlap vocabulary (same as simple_ctkd / span_ctkd)
    # ------------------------------------------------------------------
    def _find_overlap_tokens(self):
        student_vocab = {k.replace("Ġ", "▁"): v for k, v in self.student_tokenizer.get_vocab().items()}
        teacher_vocab = {k.replace("Ġ", "▁"): v for k, v in self.teacher_tokenizer.get_vocab().items()}
        overlap_tokens = set(student_vocab.keys()) & set(teacher_vocab.keys())
        student_ids = [student_vocab[token] for token in overlap_tokens]
        teacher_ids = [teacher_vocab[token] for token in overlap_tokens]
        stu_eos, tea_eos = self.student_tokenizer.eos_token_id, self.teacher_tokenizer.eos_token_id
        if stu_eos not in student_ids or tea_eos not in teacher_ids:
            student_ids.append(stu_eos)
            teacher_ids.append(tea_eos)
        device = self.teacher_lm_head.weight.device
        logger.info(f"[SpanCTKD-1to1] Num of overlap_tokens between student & teacher: {len(student_ids)}")
        return (
            torch.tensor(student_ids, dtype=torch.long, device=device),
            torch.tensor(teacher_ids, dtype=torch.long, device=device),
        )

    # ------------------------------------------------------------------
    # Alignment using tokenizer.decode (more accurate than convert_ids_to_tokens)
    # Returns only 1:1 aligned indices (like simple_ctkd's _align_sequences)
    # ------------------------------------------------------------------
    def _align_sequences(self, tea_label_ids, stu_label_ids):
        """Align teacher and student label tokens using tokenizer.decode.

        Unlike span_ctkd which returns segments (including spans), this method
        returns only 1:1 aligned position indices — same interface as
        simple_ctkd._align_sequences but with better text normalization.

        Args:
            tea_label_ids: 1-D list/tensor of teacher label token ids.
            stu_label_ids: 1-D list/tensor of student label token ids.

        Returns:
            t2s_align: list of teacher position indices that are 1:1 aligned.
            s2t_align: list of student position indices that are 1:1 aligned.
        """
        if len(tea_label_ids) == 0 or len(stu_label_ids) == 0:
            return [], []

        tea_ids_list = tea_label_ids if isinstance(tea_label_ids, list) else tea_label_ids.cpu().tolist()
        stu_ids_list = stu_label_ids if isinstance(stu_label_ids, list) else stu_label_ids.cpu().tolist()

        # Use tokenizer.decode([tid]) for accurate text normalization
        # (handles Qwen's Ċ for \n, etc.)
        tea_token_texts = [self.teacher_tokenizer.decode([tid]) for tid in tea_ids_list]
        stu_token_texts = [self.student_tokenizer.decode([tid]) for tid in stu_ids_list]

        tea_eos = self.teacher_tokenizer.eos_token
        stu_eos = self.student_tokenizer.eos_token

        i, j = 0, 0
        t2s_align = []  # teacher indices
        s2t_align = []  # student indices
        history_tea = ""
        history_stu = ""

        while i < len(tea_token_texts) and j < len(stu_token_texts):
            is_eos_match = (tea_token_texts[i] == tea_eos and stu_token_texts[j] == stu_eos)
            if history_tea == history_stu and (
                tea_token_texts[i] == stu_token_texts[j] or is_eos_match
            ):
                # 1:1 aligned position
                t2s_align.append(i)
                s2t_align.append(j)
                history_tea += tea_token_texts[i]
                history_stu += stu_token_texts[j]
                i += 1
                j += 1
            elif len(history_tea) > len(history_stu):
                history_stu += stu_token_texts[j]
                j += 1
            elif len(history_tea) < len(history_stu):
                history_tea += tea_token_texts[i]
                i += 1
            else:
                history_tea += tea_token_texts[i]
                history_stu += stu_token_texts[j]
                i += 1
                j += 1

        if self._debug_count < 3:
            self._debug_count += 1
            logger.info(
                f"[SpanCTKD-1to1 DEBUG] Found {len(t2s_align)} 1:1 aligned positions "
                f"out of tea({len(tea_token_texts)}) / stu({len(stu_token_texts)}) tokens. "
                f"tea_texts[:5]={tea_token_texts[:5]}... stu_texts[:5]={stu_token_texts[:5]}..."
            )

        return t2s_align, s2t_align

    # ------------------------------------------------------------------
    # Training step — same structure as simple_ctkd but with decode-based alignment
    # ------------------------------------------------------------------
    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_attn_mask = micro_batch["tea_attn_mask"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for KD"

        mm_kwargs = {k[3:]: v for k, v in micro_batch.items() if k.startswith("mm_")}

        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"]

        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits = self.teacher_lm_head(teacher_hiddens)

        # Flatten logits at loss_mask positions (same as simple_ctkd)
        student_logits = student_logits[student_loss_mask]

        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)[teacher_loss_mask]

        # Align using decode-based method (only 1:1 positions)
        teacher_aligned_idx, student_aligned_idx = self._align_sequences(
            teacher_label_ids.cpu().tolist(),
            student_label_ids.cpu().tolist(),
        )

        # Extract aligned logits on overlap vocabulary (same as simple_ctkd)
        aligned_student_logits = student_logits[student_aligned_idx][:, self.student_overlap_token_ids]
        aligned_teacher_logits = teacher_logits[teacher_aligned_idx][:, self.teacher_overlap_token_ids]
        assert aligned_teacher_logits.shape == aligned_student_logits.shape, \
            "teacher_logits must have the same shape with student_logits, " \
            f"but got teacher: {aligned_teacher_logits.shape} and student: {aligned_student_logits.shape}."

        align_ratio = torch.tensor(len(student_aligned_idx) / max(len(student_label_ids), 1))

        kd_loss = self.loss_fn(
            aligned_student_logits,
            aligned_teacher_logits,
            reduction="none",
        )
        kd_loss = kd_loss.sum() / avg_token_num
        loss_info = {"loss": kd_loss, "kd_loss": kd_loss, "align_ratio": align_ratio}

        if self.args.kd.kd_ratio < 1:
            ce_loss = compute_cross_entropy(student_logits, student_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
