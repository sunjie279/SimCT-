"""Span CTKD ablation: mask span segment loss (configurable ratio).

This module implements an ablation variant of SpanCrossTokenizerKD that keeps
the full virtual common vocabulary (overlap + span dimensions) but masks out
the loss contribution from a configurable fraction of span segments.  The
``span_mask_ratio`` parameter (default 1.0) controls what proportion of span
segments have their loss zeroed out:
- 1.0 = mask ALL span loss (original behaviour, only 1:1 segments contribute)
- 0.0 = mask NONE (all spans contribute, equivalent to full span_ctkd)
- 0.25/0.5/0.75 = randomly mask that fraction of span segments per sample
"""

import random

import torch

from kdflow.algorithms import register_algorithm
from kdflow.algorithms.span_ctkd import SpanCrossTokenizerKD
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


@register_algorithm("span_ctkd_no_span_loss")
class SpanCTKDNoSpanLoss(SpanCrossTokenizerKD):
    """Ablation of SpanCrossTokenizerKD: mask span segment loss.

    Inherits all alignment and virtual-vocabulary logic from the parent class.
    The only difference is in ``training_step``: after computing per-segment
    loss on the virtual common vocabulary, a configurable fraction of span
    segments (controlled by ``span_mask_ratio``) have their loss masked to
    zero via random sampling.

    When ``span_mask_ratio = 1.0`` (default), ALL span segment losses are
    masked — identical to the original behaviour.  When ``span_mask_ratio =
    0.0``, no masking is applied and all spans contribute to the loss.
    """

    # No __init__ override — fully reuse parent initialisation.

    def training_step(self, micro_batch):
        """Training step with configurable span segment loss masking.

        Identical to ``SpanCrossTokenizerKD.training_step`` except that a
        configurable fraction (``span_mask_ratio``) of span segments have
        their loss set to zero via random sampling.
        """
        span_mask_ratio = self.args.kd.span_mask_ratio
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

        # Student forward
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"]

        # Extract label ids (next-token) at loss_mask positions
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)

        # Flatten student logits at loss_mask positions
        student_logits_flat = student_logits[student_loss_mask]

        # Free full student logits early
        del student_logits, output

        # Teacher logits from hidden states (already 2D, filtered by loss_mask)
        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits_flat = self.teacher_lm_head(teacher_hiddens)
        del teacher_hiddens

        # Process each sample in the batch
        batch_size = student_input_ids.shape[0]
        total_loss = torch.tensor(0.0, device=student_logits_flat.device, requires_grad=True)
        total_aligned_tokens = 0
        total_response_tokens = 0
        total_span_segments = 0
        total_masked_spans = 0

        # Offset trackers for flattened logits
        tea_logits_offset = 0
        stu_logits_offset = 0

        for b in range(batch_size):
            stu_mask = student_loss_mask[b]
            tea_mask = teacher_loss_mask[b]

            stu_num_loss_tokens = stu_mask.sum().item()
            tea_num_loss_tokens = tea_mask.sum().item()

            # Student logits for this sample
            stu_logits_b = student_logits_flat[stu_logits_offset:stu_logits_offset + stu_num_loss_tokens]
            stu_logits_offset += stu_num_loss_tokens

            # Teacher logits for this sample
            tea_logits_b = teacher_logits_flat[tea_logits_offset:tea_logits_offset + tea_num_loss_tokens]
            tea_logits_offset += tea_num_loss_tokens

            # Extract label ids at loss_mask positions
            stu_label_ids_b = student_label_ids[b][stu_mask].cpu().tolist()
            tea_label_ids_b = teacher_label_ids[b][tea_mask].cpu().tolist()

            # Align on label ids and identify spans
            segments, tea_ids_list, stu_ids_list = self._align_sequences_with_spans(
                tea_label_ids_b, stu_label_ids_b
            )

            if len(segments) == 0:
                total_response_tokens += max(stu_num_loss_tokens, tea_num_loss_tokens)
                continue

            # Build virtual vocab logits (reuse parent method)
            stu_virtual, tea_virtual = self._build_virtual_vocab_logits(
                segments, stu_logits_b, tea_logits_b,
                stu_ids_list, tea_ids_list,
            )

            assert stu_virtual.shape == tea_virtual.shape, \
                f"Virtual logit shape mismatch: student {stu_virtual.shape} vs teacher {tea_virtual.shape}"

            # Compute RKL loss on virtual common vocabulary (per-segment)
            sample_loss = self.loss_fn(
                stu_virtual,
                tea_virtual.detach(),
                reduction="none",
            )  # [num_segments, virtual_dim] or [num_segments]

            # ---- Ablation: mask span segment loss (configurable ratio) ----
            # Identify span segment indices
            num_segments = len(segments)
            span_seg_indices = [
                seg_idx for seg_idx, (ts, te, ss, se) in enumerate(segments)
                if (te - ts) > 1 or (se - ss) > 1
            ]
            num_spans = len(span_seg_indices)
            total_span_segments += num_spans

            # Determine how many spans to mask
            num_to_mask = min(round(num_spans * span_mask_ratio), num_spans)
            if num_to_mask > 0 and num_to_mask < num_spans:
                masked_span_indices = set(random.sample(span_seg_indices, num_to_mask))
            elif num_to_mask >= num_spans:
                masked_span_indices = set(span_seg_indices)
            else:
                masked_span_indices = set()
            total_masked_spans += len(masked_span_indices)

            # Build mask: 1.0 for all segments, 0.0 for masked spans
            span_mask = torch.ones(num_segments, device=sample_loss.device, dtype=sample_loss.dtype)
            for seg_idx in masked_span_indices:
                span_mask[seg_idx] = 0.0

            # Apply mask: broadcast if sample_loss has extra dims (e.g. [num_segments, virtual_dim])
            if sample_loss.dim() > 1:
                span_mask = span_mask.unsqueeze(-1)  # [num_segments, 1]
            sample_loss = sample_loss * span_mask

            total_loss = total_loss + sample_loss.sum()

            # Count aligned tokens that actually contribute to loss
            for seg_idx, (ts, te, ss, se) in enumerate(segments):
                if (te - ts) == 1 and (se - ss) == 1:
                    # 1:1 aligned segment — always contributes
                    total_aligned_tokens += 1
                elif seg_idx not in masked_span_indices:
                    # Unmasked span segment — also contributes
                    total_aligned_tokens += max(te - ts, se - ss)
            total_response_tokens += max(stu_num_loss_tokens, tea_num_loss_tokens)

        # Normalize
        kd_loss = total_loss / avg_token_num

        align_ratio = torch.tensor(
            total_aligned_tokens / max(total_response_tokens, 1),
            device=student_logits_flat.device,
        )

        span_mask_ratio_actual = torch.tensor(
            total_masked_spans / max(total_span_segments, 1),
            device=student_logits_flat.device,
        )

        loss_info = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "align_ratio": align_ratio,
            "span_mask_ratio_actual": span_mask_ratio_actual,
        }

        if self.args.kd.kd_ratio < 1:
            ce_label_ids = student_label_ids[student_loss_mask]
            ce_loss = compute_cross_entropy(student_logits_flat, ce_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
