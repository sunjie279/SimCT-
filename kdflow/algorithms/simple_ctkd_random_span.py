"""Simple CTKD with Random Span Merging — Ablation Experiment.

This module implements an ablation variant of SimpleCrossTokenizerKD that
randomly merges a configurable fraction of 1:1-aligned token positions into
spans, then computes KD loss on a virtual common vocabulary (overlap tokens +
span dimensions) following the same logic as SpanCrossTokenizerKD.

Purpose: demonstrate that span-based virtual tokens are a good approximation
of individual tokens at low merge ratios, but degrade when the merge ratio
becomes too high (e.g. 10% vs 30% vs 50% vs 70%).
"""

import random

import torch
import torch.nn.functional as F

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


@register_algorithm("simple_ctkd_random_span")
class SimpleCTKDRandomSpan:
    """Ablation: randomly merge aligned tokens into spans in simple_ctkd.

    Flow:
    1. Find overlap vocabulary (same as simple_ctkd).
    2. Align teacher & student label sequences to get 1:1 positions.
    3. Randomly mark ``random_span_ratio`` fraction of aligned positions as
       "to-merge", then group consecutive marked positions (plus the next
       unmarked position as anchor) into span segments.
    4. Build virtual common vocabulary logits (overlap + span dims) following
       span_ctkd logic.
    5. Compute KD loss on the virtual vocabulary.

    Memory-optimised: follows simple_ctkd's flat processing pattern — no
    per-sample loop, immediate overlap-subset slicing after alignment so that
    full-vocab logits are released early.
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
        self.student_overlap_token_ids, self.teacher_overlap_token_ids = (
            self._find_overlap_tokens()
        )
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
        self.random_span_ratio = self.args.kd.random_span_ratio
        self._debug_count = 0
        self._span_debug_count = 0
        logger.info(
            f"[SimpleCTKD-RandomSpan] random_span_ratio={self.random_span_ratio}, "
            f"num_overlap_tokens={self.student_overlap_token_ids.shape[0]}"
        )

    # ------------------------------------------------------------------
    # Overlap vocabulary (same as simple_ctkd)
    # ------------------------------------------------------------------
    def _find_overlap_tokens(self):
        """Find the overlap tokens between student and teacher tokenizer."""
        student_vocab = {
            k.replace("Ġ", "▁"): v
            for k, v in self.student_tokenizer.get_vocab().items()
        }
        teacher_vocab = {
            k.replace("Ġ", "▁"): v
            for k, v in self.teacher_tokenizer.get_vocab().items()
        }
        overlap_tokens = set(student_vocab.keys()) & set(teacher_vocab.keys())
        student_ids = [student_vocab[token] for token in overlap_tokens]
        teacher_ids = [teacher_vocab[token] for token in overlap_tokens]
        stu_eos = self.student_tokenizer.eos_token_id
        tea_eos = self.teacher_tokenizer.eos_token_id
        if stu_eos not in student_ids or tea_eos not in teacher_ids:
            student_ids.append(stu_eos)
            teacher_ids.append(tea_eos)
        device = self.teacher_lm_head.weight.device
        logger.info(
            f"[SimpleCTKD-RandomSpan] Num of overlap_tokens between "
            f"student & teacher: {len(student_ids)}"
        )
        return (
            torch.tensor(student_ids, dtype=torch.long, device=device),
            torch.tensor(teacher_ids, dtype=torch.long, device=device),
        )

    # ------------------------------------------------------------------
    # Sequence alignment (same greedy algorithm as simple_ctkd)
    # ------------------------------------------------------------------
    def _align_sequences(self, tea_seq, stu_seq):
        """Align teacher and student token sequences, returning 1:1 indices.

        Args:
            tea_seq: list of teacher token strings (from convert_ids_to_tokens,
                     already filtered by loss_mask).
            stu_seq: list of student token strings.

        Returns:
            t2s_align: list of teacher position indices (into loss_mask seq).
            s2t_align: list of student position indices (into loss_mask seq).
        """
        i, j = 0, 0
        t2s_align, s2t_align = [], []
        history_tea_seq, history_stu_seq = "", ""

        tea_eos = self.teacher_tokenizer.eos_token
        stu_eos = self.student_tokenizer.eos_token

        tea_seq = [token.replace("▁", "").replace("Ġ", "") for token in tea_seq]
        stu_seq = [token.replace("▁", "").replace("Ġ", "") for token in stu_seq]

        if tea_seq == stu_seq:
            indices = list(range(len(tea_seq)))
            return indices, indices

        while i < len(tea_seq) and j < len(stu_seq):
            is_eos_match = tea_seq[i] == tea_eos and stu_seq[j] == stu_eos
            if history_tea_seq == history_stu_seq and (
                tea_seq[i] == stu_seq[j] or is_eos_match
            ):
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

    # ------------------------------------------------------------------
    # Random span merging
    # ------------------------------------------------------------------
    def _random_merge_to_spans(self, tea_aligned_idx, stu_aligned_idx):
        """Randomly merge aligned positions into span segments.

        Given the 1:1 aligned position lists from ``_align_sequences``, this
        method randomly selects ``ratio`` fraction of positions to mark as
        "to-merge".  It then scans left-to-right and groups consecutive marked
        positions together with the next unmarked position (anchor) to form a
        span segment.  Unmarked positions that are not consumed as anchors
        remain as single-token segments.

        The ``random_span_ratio`` controls the fraction of aligned tokens that
        end up in a *merged* (non-singleton) state.

        Args:
            tea_aligned_idx: list of teacher indices (into loss_mask sequence).
            stu_aligned_idx: list of student indices (into loss_mask sequence).

        Returns:
            segments: list of (tea_indices, stu_indices) tuples where each
                element is a list of position indices forming one segment.
                Single-token segments have length 1; span segments have
                length > 1.
        """
        n = len(tea_aligned_idx)
        if n == 0:
            return []

        ratio = self.random_span_ratio
        if ratio <= 0.0:
            # No merging — every position is a single-token segment
            return [
                ([tea_aligned_idx[k]], [stu_aligned_idx[k]]) for k in range(n)
            ]

        num_to_merge = min(round(n * ratio), n)
        if num_to_merge == 0:
            return [
                ([tea_aligned_idx[k]], [stu_aligned_idx[k]]) for k in range(n)
            ]

        # Randomly select positions to mark as "to-merge"
        merge_set = set(random.sample(range(n), num_to_merge))

        segments = []
        k = 0
        actual_merged_tokens = 0  # tokens that end up in multi-token spans

        while k < n:
            if k not in merge_set:
                # Unmarked position → single-token segment
                segments.append(
                    ([tea_aligned_idx[k]], [stu_aligned_idx[k]])
                )
                k += 1
            else:
                # Start of a span: collect consecutive marked positions
                span_tea = []
                span_stu = []
                while k < n and k in merge_set:
                    span_tea.append(tea_aligned_idx[k])
                    span_stu.append(stu_aligned_idx[k])
                    k += 1
                # Append the next unmarked position as anchor (if exists)
                if k < n:
                    span_tea.append(tea_aligned_idx[k])
                    span_stu.append(stu_aligned_idx[k])
                    k += 1
                # All tokens in this span are "merged"
                actual_merged_tokens += len(span_tea)
                segments.append((span_tea, span_stu))

        # Debug logging for first few steps
        if self._debug_count < 3:
            num_spans = sum(1 for seg in segments if len(seg[0]) > 1)
            span_token_count = sum(
                len(seg[0]) for seg in segments if len(seg[0]) > 1
            )
            logger.info(
                f"[SimpleCTKD-RandomSpan DEBUG] "
                f"total_aligned={n}, num_to_merge={num_to_merge}, "
                f"num_spans={num_spans}, span_token_count={span_token_count}, "
                f"actual_merged_ratio={actual_merged_tokens / n:.3f}, "
                f"num_segments={len(segments)}"
            )

        return segments

    # ------------------------------------------------------------------
    # Build virtual common vocabulary logits (memory-optimised)
    # ------------------------------------------------------------------
    def _build_virtual_vocab_logits(
        self,
        segments,
        stu_overlap_logits,
        tea_overlap_logits,
        stu_self_logits,
        tea_self_logits,
    ):
        """Build student and teacher logit matrices on the virtual common vocab.

        Memory-optimised version: works on **overlap-subset logits** and
        **self-logits** instead of full-vocabulary logits.

        For each segment we construct a logit vector of size
        ``num_overlap + num_total_spans`` where:
        - the first ``num_overlap`` dims come from the base overlap vocabulary,
        - the remaining ``num_total_spans`` dims correspond to span logits.

        For a 1:1 segment the overlap-vocab logits are taken directly from the
        single token.  For a span the overlap-vocab logits are taken from the
        **first** token position only.  The span-logit dimension for the
        current span gets the mean of each constituent token's self-logit.

        Args:
            segments: list of (tea_indices, stu_indices) — each element is a
                      pair of lists of **local** position indices (0-based into
                      the aligned arrays).
            stu_overlap_logits: [num_aligned, num_overlap] — student logits on
                                the overlap vocabulary subset.
            tea_overlap_logits: [num_aligned, num_overlap] — teacher logits on
                                the overlap vocabulary subset.
            stu_self_logits: [num_aligned] — each position's logit at its own
                             label token id (student side).
            tea_self_logits: [num_aligned] — each position's logit at its own
                             label token id (teacher side).

        Returns:
            stu_virtual_logits: [num_segments, num_overlap + num_spans]
            tea_virtual_logits: [num_segments, num_overlap + num_spans]
        """
        num_overlap = stu_overlap_logits.shape[1]
        device = stu_overlap_logits.device

        # Identify span segments (multi-token)
        span_indices = []
        for seg_idx, (tea_idx_list, stu_idx_list) in enumerate(segments):
            if len(tea_idx_list) > 1 or len(stu_idx_list) > 1:
                span_indices.append(seg_idx)
        num_spans = len(span_indices)

        if num_spans == 0:
            # Fast path: no spans at all — just gather overlap logits for each
            # segment's first position.
            first_positions = torch.tensor(
                [seg[1][0] for seg in segments],  # stu first pos
                dtype=torch.long,
                device=device,
            )
            stu_virtual = stu_overlap_logits[first_positions]
            tea_first_positions = torch.tensor(
                [seg[0][0] for seg in segments],  # tea first pos
                dtype=torch.long,
                device=device,
            )
            tea_virtual = tea_overlap_logits[tea_first_positions]
            return stu_virtual, tea_virtual

        # There are spans — build the full virtual vocab
        seg_to_span_dim = {
            seg_idx: dim_idx for dim_idx, seg_idx in enumerate(span_indices)
        }

        num_segments = len(segments)
        virtual_dim = num_overlap + num_spans

        # Pre-allocate output tensors
        stu_virtual = torch.empty(
            (num_segments, virtual_dim), device=device, dtype=stu_overlap_logits.dtype
        )
        tea_virtual = torch.empty(
            (num_segments, virtual_dim), device=device, dtype=tea_overlap_logits.dtype
        )

        # Fill span dims with -1e9 (negligible after softmax)
        stu_virtual[:, num_overlap:] = -1e9
        tea_virtual[:, num_overlap:] = -1e9

        # Gather first-position indices for all segments (vectorised)
        stu_first_pos = torch.tensor(
            [seg[1][0] for seg in segments], dtype=torch.long, device=device
        )
        tea_first_pos = torch.tensor(
            [seg[0][0] for seg in segments], dtype=torch.long, device=device
        )

        # Fill overlap dims for all segments at once (vectorised)
        stu_virtual[:, :num_overlap] = stu_overlap_logits[stu_first_pos]
        tea_virtual[:, :num_overlap] = tea_overlap_logits[tea_first_pos]

        # Fill span dims only for span segments (small loop — only over spans)
        for seg_idx in span_indices:
            dim_pos = seg_to_span_dim[seg_idx]
            tea_idx_list, stu_idx_list = segments[seg_idx]

            # Student span logit: mean of each token's self-logit
            stu_positions = torch.tensor(stu_idx_list, dtype=torch.long, device=device)
            stu_span_val = stu_self_logits[stu_positions].mean()
            stu_virtual[seg_idx, num_overlap + dim_pos] = stu_span_val

            # Teacher span logit: mean of each token's self-logit
            tea_positions = torch.tensor(tea_idx_list, dtype=torch.long, device=device)
            tea_span_val = tea_self_logits[tea_positions].mean()
            tea_virtual[seg_idx, num_overlap + dim_pos] = tea_span_val

            # Debug logging for first span
            if self._span_debug_count < 1:
                self._span_debug_count += 1
                logger.info(
                    f"\n[SimpleCTKD-RandomSpan SPAN DEBUG] "
                    f"seg_idx={seg_idx}, span_len={len(stu_idx_list)}\n"
                    f"  Student local positions: {stu_idx_list}\n"
                    f"  Student self_logits: "
                    f"{stu_self_logits[stu_positions].detach().cpu().tolist()}\n"
                    f"  Student span_dim (mean): {stu_span_val.item():.4f}\n"
                    f"  Teacher local positions: {tea_idx_list}\n"
                    f"  Teacher self_logits: "
                    f"{tea_self_logits[tea_positions].detach().cpu().tolist()}\n"
                    f"  Teacher span_dim (mean): {tea_span_val.item():.4f}\n"
                    f"  num_overlap={num_overlap}, num_spans={num_spans}"
                )

        return stu_virtual, tea_virtual

    # ------------------------------------------------------------------
    # Training step (memory-optimised, simple_ctkd-style flat processing)
    # ------------------------------------------------------------------
    def training_step(self, micro_batch):
        """One training step with random-span cross-tokenizer KD loss.

        Memory-optimised flow (follows simple_ctkd pattern):
        1. Student forward → full-sequence logits.
        2. Teacher logits from hidden states (loss_mask positions only).
        3. Flatten logits at loss_mask positions.
        4. Align sequences (1:1 positions).
        5. **Immediately** slice to overlap subset + extract self-logits,
           then delete full-vocab logits to free GPU memory.
        6. Randomly merge aligned positions into spans.
        7. Build virtual vocab logits on overlap + self-logits (no full vocab).
        8. Compute KD loss.
        """
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_attn_mask = micro_batch["tea_attn_mask"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, (
            "micro_batch must contain `teacher_hiddens` for KD"
        )

        mm_kwargs = {
            k[3:]: v for k, v in micro_batch.items() if k.startswith("mm_")
        }

        # ---- Student forward ----
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"]

        # Extract label ids (next-token)
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)

        # Flatten student logits at loss_mask positions
        student_logits_flat = student_logits[student_loss_mask]  # [N_stu, vocab_s]
        del student_logits, output

        # ---- CE loss (compute BEFORE deleting student_logits_flat) ----
        ce_loss = None
        if self.args.kd.kd_ratio < 1:
            ce_label_ids = student_label_ids[student_loss_mask]
            ce_loss = (
                compute_cross_entropy(
                    student_logits_flat, ce_label_ids, reduction="sum"
                )
                / avg_token_num
            )

        # ---- Teacher logits ----
        # teacher_hiddens is already 2D [total_tea_loss_tokens, hidden_size]
        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits_flat = self.teacher_lm_head(teacher_hiddens)  # [N_tea, vocab_t]
        del teacher_hiddens

        # ---- Label ids at loss_mask positions (flat, for alignment) ----
        student_label_ids_flat = student_label_ids[student_loss_mask]  # [N_stu]
        teacher_label_ids_flat = teacher_label_ids[teacher_loss_mask]  # [N_tea]

        stu_label_ids_list = student_label_ids_flat.cpu().tolist()
        tea_label_ids_list = teacher_label_ids_flat.cpu().tolist()

        # ---- Sequence alignment (1:1 positions) ----
        teacher_aligned_idx, student_aligned_idx = self._align_sequences(
            self.teacher_tokenizer.convert_ids_to_tokens(tea_label_ids_list),
            self.student_tokenizer.convert_ids_to_tokens(stu_label_ids_list),
        )

        num_aligned = len(teacher_aligned_idx)
        total_response_tokens = max(len(stu_label_ids_list), len(tea_label_ids_list))

        if num_aligned == 0:
            # No aligned tokens — return zero KD loss
            device = student_logits_flat.device
            del student_logits_flat, teacher_logits_flat
            kd_loss = torch.tensor(0.0, device=device, requires_grad=True)
            align_ratio = torch.tensor(0.0, device=device)
            span_ratio = torch.tensor(0.0, device=device)
            loss_info = {
                "loss": kd_loss,
                "kd_loss": kd_loss,
                "align_ratio": align_ratio,
                "span_ratio": span_ratio,
            }
            if ce_loss is not None:
                loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
                loss_info["loss"] = loss
                loss_info["ce_loss"] = ce_loss
            return loss_info

        # ---- Extract aligned logits (still full vocab) ----
        aligned_stu_logits = student_logits_flat[student_aligned_idx]  # [num_aligned, vocab_s]
        aligned_tea_logits = teacher_logits_flat[teacher_aligned_idx]  # [num_aligned, vocab_t]

        # Free full-vocab flat logits immediately
        device = student_logits_flat.device
        del student_logits_flat, teacher_logits_flat

        # ---- Slice to overlap subset (key memory optimisation) ----
        stu_overlap_logits = aligned_stu_logits[:, self.student_overlap_token_ids]  # [num_aligned, num_overlap]
        tea_overlap_logits = aligned_tea_logits[:, self.teacher_overlap_token_ids]  # [num_aligned, num_overlap]

        # ---- Extract self-logits (each position's logit at its own label id) ----
        # These are needed for span logit computation.
        aligned_stu_label_ids = torch.tensor(
            [stu_label_ids_list[i] for i in student_aligned_idx],
            dtype=torch.long,
            device=device,
        )
        aligned_tea_label_ids = torch.tensor(
            [tea_label_ids_list[i] for i in teacher_aligned_idx],
            dtype=torch.long,
            device=device,
        )

        # Gather self-logits: logit[i, label_id[i]] for each aligned position
        stu_self_logits = aligned_stu_logits[
            torch.arange(num_aligned, device=device), aligned_stu_label_ids
        ]  # [num_aligned]
        tea_self_logits = aligned_tea_logits[
            torch.arange(num_aligned, device=device), aligned_tea_label_ids
        ]  # [num_aligned]

        # Free aligned full-vocab logits — only overlap + self-logits remain
        del aligned_stu_logits, aligned_tea_logits

        # ---- Random span merging ----
        # _random_merge_to_spans expects indices into the loss_mask sequence,
        # but we need LOCAL indices (0-based into the aligned arrays) for
        # indexing into stu_overlap_logits / stu_self_logits.
        # We pass range(num_aligned) as both tea and stu aligned indices so
        # that the returned segments contain local indices directly.
        local_tea_idx = list(range(num_aligned))
        local_stu_idx = list(range(num_aligned))
        segments = self._random_merge_to_spans(local_tea_idx, local_stu_idx)

        if len(segments) == 0:
            kd_loss = torch.tensor(0.0, device=device, requires_grad=True)
            align_ratio = torch.tensor(
                num_aligned / max(total_response_tokens, 1), device=device
            )
            span_ratio = torch.tensor(0.0, device=device)
            loss_info = {
                "loss": kd_loss,
                "kd_loss": kd_loss,
                "align_ratio": align_ratio,
                "span_ratio": span_ratio,
            }
            if ce_loss is not None:
                loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
                loss_info["loss"] = loss
                loss_info["ce_loss"] = ce_loss
            return loss_info

        # Count merged tokens (tokens in multi-token spans)
        total_merged_tokens = sum(
            len(seg[0]) for seg in segments if len(seg[0]) > 1
        )

        # ---- Build virtual vocab logits (on overlap + self-logits) ----
        stu_virtual, tea_virtual = self._build_virtual_vocab_logits(
            segments,
            stu_overlap_logits,
            tea_overlap_logits,
            stu_self_logits,
            tea_self_logits,
        )

        assert stu_virtual.shape == tea_virtual.shape, (
            f"Virtual logit shape mismatch: "
            f"student {stu_virtual.shape} vs teacher {tea_virtual.shape}"
        )

        # ---- Compute KD loss ----
        kd_loss_raw = self.loss_fn(
            stu_virtual,
            tea_virtual.detach(),
            reduction="none",
        )
        kd_loss = kd_loss_raw.sum() / avg_token_num

        # Increment debug counter
        self._debug_count += 1

        # ---- Metrics ----
        align_ratio = torch.tensor(
            num_aligned / max(total_response_tokens, 1), device=device
        )
        span_ratio = torch.tensor(
            total_merged_tokens / max(num_aligned, 1), device=device
        )

        loss_info = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "align_ratio": align_ratio,
            "span_ratio": span_ratio,
        }

        if ce_loss is not None:
            loss = (
                (1 - self.args.kd.kd_ratio) * ce_loss
                + self.args.kd.kd_ratio * kd_loss
            )
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
