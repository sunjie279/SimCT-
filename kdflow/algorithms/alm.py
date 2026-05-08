import torch
import torch.nn.functional as F

from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)


@register_algorithm("alm")
class ALM:
    """Approximate Likelihood Matching (ALM) for cross-tokenizer distillation.
    
    Reference: "Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching"
    (Minixhofer et al., NeurIPS 2025)
    
    Core idea: Find aligned chunks of tokens between teacher and student sequences
    (chunks encoding the same text), then minimize a binarised f-divergence between
    their chunk-level log-probabilities. This enables distillation across fundamentally
    different tokenizers.
    
    Adapted to on-policy KDFlow framework where teacher hidden states are provided
    in each micro_batch.
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
        
        # ALM hyperparameters
        self.alm_temperature = getattr(self.args.kd, "alm_temperature", 100.0)
        self.alm_f_divergence = getattr(self.args.kd, "alm_f_divergence", "kl")
        self.alm_debiasing = getattr(self.args.kd, "alm_debiasing", False)
        self.alm_debiasing_threshold = getattr(self.args.kd, "alm_debiasing_threshold", 0.1)
        
        self._debug_count = 0  # Counter for debug logging
        
        logger.info(
            f"[ALM] Initialized with temperature={self.alm_temperature}, "
            f"f_divergence={self.alm_f_divergence}, "
            f"debiasing={self.alm_debiasing}"
        )
    
    def _compute_chunk_alignment(self, tea_input_ids, stu_input_ids, tea_loss_mask, stu_loss_mask):
        """Compute aligned chunks between teacher and student token sequences.
        
        Uses tokenizer.decode() for each individual token to get the true text
        representation, then performs cumulative text comparison to find chunk
        boundaries. This correctly handles all tokenizer conventions (byte-level
        BPE, SentencePiece, etc.) including special characters like newlines.
        
        NOTE on index semantics (critical for correctness):
        loss_mask follows the standard next-token-prediction convention where
        ``loss_mask[i] = True`` means ``logits[i]`` is used to predict
        ``input_ids[i+1]``. So the first True position (``loss_indices[0]``) is
        actually the *last prompt token*, NOT the first response token.
        The tokens being *predicted* (i.e. the actual response tokens + EOS)
        live at positions ``loss_indices[0]+1 .. loss_indices[-1]+1`` inclusive.
        
        Alignment must be performed on these *predicted* response tokens so that
        the compared text is identical on teacher and student side (both rolled
        out the same ``response_text``). Chunks are returned with ``[start, end)``
        indices in *logits-position* space, i.e. they are valid indices into the
        teacher's ``loss_indices`` / into the shifted student ``token_log_probs``.
        
        Args:
            tea_input_ids: Teacher input token ids [seq_len_t]
            stu_input_ids: Student input token ids [seq_len_s]
            tea_loss_mask: Teacher loss mask [seq_len_t]
            stu_loss_mask: Student loss mask [seq_len_s]
            
        Returns:
            List of tuples (tea_chunk_start, tea_chunk_end, stu_chunk_start, stu_chunk_end)
            representing aligned chunks in *logits-position* space of the full sequence.
        """
        tea_loss_indices = tea_loss_mask.nonzero(as_tuple=True)[0]
        stu_loss_indices = stu_loss_mask.nonzero(as_tuple=True)[0]
        
        if len(tea_loss_indices) == 0 or len(stu_loss_indices) == 0:
            return []
        
        # Range of *predicted* response tokens (exclude the last-prompt-token
        # that sits at loss_indices[0]; include the final EOS which is predicted
        # by logits at loss_indices[-1]).
        tea_pred_start = tea_loss_indices[0].item() + 1
        tea_pred_end = tea_loss_indices[-1].item() + 2
        stu_pred_start = stu_loss_indices[0].item() + 1
        stu_pred_end = stu_loss_indices[-1].item() + 2
        
        tea_resp_ids = tea_input_ids[tea_pred_start:tea_pred_end].cpu().tolist()
        stu_resp_ids = stu_input_ids[stu_pred_start:stu_pred_end].cpu().tolist()
        
        if len(tea_resp_ids) == 0 or len(stu_resp_ids) == 0:
            return []
        
        # Use tokenizer.decode() for each token to get true text representation.
        # This correctly handles all tokenizer conventions (byte-level BPE special
        # chars like Ċ for newline, SentencePiece ▁ for space, etc.)
        # O(n) decode calls per sequence - fast enough for ~160 tokens.
        EOS_MARKER = "<|EOS|>"
        
        def decode_tokens(ids_list, tokenizer):
            """Decode each token individually to get its text contribution."""
            eos_id = tokenizer.eos_token_id
            texts = []
            for tid in ids_list:
                if tid == eos_id:
                    texts.append(EOS_MARKER)
                elif tid in set(tokenizer.all_special_ids):
                    texts.append("")  # Skip non-EOS special tokens
                else:
                    texts.append(tokenizer.decode([tid]))
            return texts
        
        tea_token_texts = decode_tokens(tea_resp_ids, self.teacher_tokenizer)
        stu_token_texts = decode_tokens(stu_resp_ids, self.student_tokenizer)
        
        # Find alignment boundaries using cumulative text comparison
        # Same greedy algorithm as simple_ctkd._align_sequences
        i, j = 0, 0
        boundaries = []
        history_tea = ""
        history_stu = ""
        
        while i < len(tea_token_texts) and j < len(stu_token_texts):
            if history_tea == history_stu and tea_token_texts[i] == stu_token_texts[j]:
                boundaries.append((i, j))
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
        
        if len(boundaries) == 0:
            # Debug: log details for first few failures
            if self._debug_count < 3:
                self._debug_count += 1
                logger.warning(
                    f"[ALM DEBUG] No boundaries found! "
                    f"tea_token_texts({len(tea_token_texts)}): {tea_token_texts[:10]}... "
                    f"stu_token_texts({len(stu_token_texts)}): {stu_token_texts[:10]}... "
                    f"tea_pred_range=[{tea_pred_start},{tea_pred_end}) "
                    f"stu_pred_range=[{stu_pred_start},{stu_pred_end})"
                )
            return []
        
        # Debug: log success for first few calls
        if self._debug_count < 3:
            self._debug_count += 1
            logger.info(
                f"[ALM DEBUG] Found {len(boundaries)} boundaries! "
                f"tea_texts({len(tea_token_texts)}): {tea_token_texts[:5]}... "
                f"stu_texts({len(stu_token_texts)}): {stu_token_texts[:5]}..."
            )
        
        # Convert boundaries to chunks. We return indices in *logits-position*
        # space, i.e. the position k where ``logits[k]`` predicts the token.
        # Since predicted-token position p corresponds to logits position p-1,
        # and tea_resp_ids[local] lives at global position ``tea_pred_start + local``,
        # the matching logits position is ``tea_pred_start + local - 1
        # = tea_loss_indices[0] + local``.
        tea_logits_base = tea_loss_indices[0].item()
        stu_logits_base = stu_loss_indices[0].item()
        chunks = []
        for idx in range(len(boundaries)):
            if idx == 0:
                local_tea_start, local_stu_start = 0, 0
            else:
                local_tea_start = boundaries[idx - 1][0] + 1
                local_stu_start = boundaries[idx - 1][1] + 1
            local_tea_end = boundaries[idx][0] + 1
            local_stu_end = boundaries[idx][1] + 1
            
            global_tea_start = tea_logits_base + local_tea_start
            global_tea_end = tea_logits_base + local_tea_end
            global_stu_start = stu_logits_base + local_stu_start
            global_stu_end = stu_logits_base + local_stu_end
            
            chunks.append((global_tea_start, global_tea_end, global_stu_start, global_stu_end))
        
        return chunks
    
    def _compute_chunk_log_probs_full_seq(self, logits, input_ids, chunks):
        """Compute chunk-level log-probabilities from full-sequence logits.
        
        Used for student side where logits cover the entire sequence.
        
        Chunks are given in *logits-position* space: ``logits[k]`` predicts
        ``input_ids[k+1]``. Chunk ``[start, end)`` accumulates
        ``log p(input_ids[start+1:end+1])``.
        
        Memory optimization: instead of computing log_softmax over the entire
        sequence, we only extract the positions needed by chunks and compute
        log_softmax on those positions only.
        
        Args:
            logits: Model logits [seq_len, vocab_size]
            input_ids: Input token ids [seq_len]
            chunks: List of (start, end) tuples (logits-position indices)
            
        Returns:
            Tensor of chunk log-probabilities [num_chunks]
        """
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        
        if len(chunks) == 0:
            return torch.tensor([], device=logits.device)
        
        # Collect all positions needed by chunks to avoid full-sequence log_softmax
        all_positions = []
        for start, end in chunks:
            all_positions.extend(range(start, end))
        
        if len(all_positions) == 0:
            return torch.tensor([], device=logits.device)
        
        positions_tensor = torch.tensor(all_positions, dtype=torch.long, device=logits.device)
        
        # Only extract and compute log_softmax on needed positions
        needed_logits = logits[positions_tensor]  # [num_needed, vocab_size]
        log_probs = F.log_softmax(needed_logits.float(), dim=-1)
        
        # Get the labels for these positions (shifted by 1)
        shifted_labels = input_ids[positions_tensor + 1]  # [num_needed]
        token_log_probs = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        
        # Split token_log_probs back into chunks and sum
        chunk_log_probs = []
        offset = 0
        for start, end in chunks:
            chunk_len = end - start
            chunk_lp = token_log_probs[offset:offset + chunk_len].sum()
            chunk_log_probs.append(chunk_lp)
            offset += chunk_len
        
        return torch.stack(chunk_log_probs)
    
    def _compute_chunk_log_probs_loss_region(self, logits, input_ids, loss_mask, chunks):
        """Compute chunk-level log-probabilities from loss-region-only logits.
        
        Used for teacher side where logits only cover loss_mask=True positions.
        Teacher hidden states (and thus logits) are already filtered by loss_mask
        in the SGLang engine, so ``logits[k]`` corresponds to the k-th True
        position ``loss_indices[k]`` in the full sequence; it predicts
        ``input_ids[loss_indices[k] + 1]``.
        
        Chunks are given in *logits-position* space (full-sequence indices k
        where ``loss_mask[k] == True``). We map them into the compact
        ``logits`` tensor via ``global_to_local``.
        
        Args:
            logits: Teacher logits [num_loss_tokens, vocab_size]
            input_ids: Full input token ids [seq_len]
            loss_mask: Boolean loss mask [seq_len]
            chunks: List of (start, end) tuples (logits-position indices in full sequence)
            
        Returns:
            Tensor of chunk log-probabilities [num_chunks]
        """
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        
        # Get the positions where loss_mask is True
        loss_indices = loss_mask.nonzero(as_tuple=True)[0]  # [num_loss_tokens]
        num_loss_tokens = len(loss_indices)
        
        # token_log_probs[k] = log p(input_ids[loss_indices[k] + 1] | prefix)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        shifted_labels = input_ids[loss_indices + 1]
        shifted_labels = shifted_labels[:log_probs.shape[0]]
        token_log_probs = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        
        # Build a mapping: full-sequence position -> local index in compact logits
        seq_len = input_ids.shape[0]
        global_to_local = torch.full((seq_len,), -1, dtype=torch.long, device=logits.device)
        global_to_local[loss_indices] = torch.arange(num_loss_tokens, device=logits.device)
        
        chunk_log_probs = []
        for start, end in chunks:
            local_indices = global_to_local[start:end]
            valid = (local_indices >= 0) & (local_indices < len(token_log_probs))
            if valid.any():
                chunk_lp = token_log_probs[local_indices[valid]].sum()
            else:
                chunk_lp = torch.tensor(0.0, device=logits.device)
            chunk_log_probs.append(chunk_lp)
        
        if len(chunk_log_probs) == 0:
            return torch.tensor([], device=logits.device)
        
        return torch.stack(chunk_log_probs)
    
    def _binarised_f_divergence(self, log_p_teacher, log_p_student):
        """Compute binarised f-divergence between chunk probabilities.
        
        The binarised f-divergence is:
        D_f^bin(p_T || p_S) = f(p_T^{1/tau} || p_S^{1/tau}) + f(1 - p_T^{1/tau} || 1 - p_S^{1/tau})
        
        For KL divergence: f(p || q) = p * log(p/q)
        For TVD: f(p || q) = |p - q|
        
        When tau -> infinity, the KL case simplifies to:
        C * (log_p - log_q) + C * log_p * log(log_q / log_p)
        And TVD simplifies to: C * |log_p - log_q|
        
        Args:
            log_p_teacher: Teacher chunk log-probabilities [num_chunks]
            log_p_student: Student chunk log-probabilities [num_chunks]
            
        Returns:
            Scalar loss value
        """
        tau = self.alm_temperature
        
        if self.alm_f_divergence == "tvd":
            if tau >= 50.0:
                # Use the closed-form approximation for tau -> infinity
                # f_TVD ≈ C * |log_p - log_q| (both terms are equal)
                loss = 2.0 * torch.abs(log_p_teacher - log_p_student)
            else:
                # Compute p^{1/tau} = exp(log_p / tau)
                p_t = torch.exp(log_p_teacher / tau)
                p_s = torch.exp(log_p_student / tau)
                loss = torch.abs(p_t - p_s) + torch.abs((1 - p_t) - (1 - p_s))
        elif self.alm_f_divergence == "kl":
            if tau >= 50.0:
                # Use the closed-form approximation for tau -> infinity
                # Term 1: C * (log_p_T - log_p_S)
                term1 = log_p_teacher - log_p_student
                # Term 2: C * log_p_T * log(log_p_S / log_p_T)
                # Note: log_p values are negative, so we need care with the ratio
                eps = 1e-8
                log_p_t_safe = log_p_teacher.clamp(max=-eps)
                log_p_s_safe = log_p_student.clamp(max=-eps)
                term2 = log_p_t_safe * torch.log(log_p_s_safe / log_p_t_safe)
                loss = term1 + term2
            else:
                # Compute p^{1/tau} = exp(log_p / tau)
                p_t = torch.exp(log_p_teacher / tau).clamp(1e-8, 1 - 1e-8)
                p_s = torch.exp(log_p_student / tau).clamp(1e-8, 1 - 1e-8)
                # f_KL(p_T || p_S) = p_T * log(p_T / p_S)
                kl_pos = p_t * torch.log(p_t / p_s)
                # f_KL(1 - p_T || 1 - p_S)
                kl_neg = (1 - p_t) * torch.log((1 - p_t) / (1 - p_s))
                loss = kl_pos + kl_neg
        else:
            raise ValueError(f"Unknown f-divergence: {self.alm_f_divergence}")
        
        return loss
    
    def training_step(self, micro_batch):
        """Perform one training step with ALM loss.
        
        The on-policy flow:
        1. Student generates responses (rollout)
        2. Teacher computes hidden states on the same text
        3. Both student and teacher logits are computed
        4. Aligned chunks are found between the two token sequences
        5. Chunk-level log-probabilities are computed
        6. Binarised f-divergence loss is minimized
        """
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_attn_mask = micro_batch["tea_attn_mask"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]
        
        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for ALM"
        
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
        
        # Process each sample in the batch
        batch_size = student_input_ids.shape[0]
        total_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        total_chunks = 0
        total_possible_chunks = 0
        
        # Teacher logits are 2D [num_loss_tokens_total, vocab_t] (only loss_mask positions)
        # We need to track the offset into teacher_logits for each sample
        tea_logits_offset = 0
        
        for b in range(batch_size):
            # Get per-sample data
            stu_ids = student_input_ids[b]
            tea_ids = teacher_input_ids[b]
            stu_mask = student_loss_mask[b]
            tea_mask = teacher_loss_mask[b]
            
            # Teacher logits: only loss_mask=True positions
            # teacher_logits is [total_loss_tokens_across_batch, vocab_t]
            tea_num_loss_tokens = tea_mask.sum().item()
            tea_logits_b = teacher_logits[tea_logits_offset:tea_logits_offset + tea_num_loss_tokens]
            tea_logits_offset += tea_num_loss_tokens
            
            # Compute chunk alignment (returns global indices in full sequence)
            chunks = self._compute_chunk_alignment(tea_ids, stu_ids, tea_mask, stu_mask)
            
            if len(chunks) == 0:
                continue
            
            # Separate teacher and student chunk indices
            tea_chunks = [(c[0], c[1]) for c in chunks]
            stu_chunks = [(c[2], c[3]) for c in chunks]
            
            # Compute chunk-level log-probabilities
            # Teacher: logits only cover loss_mask region
            tea_chunk_log_probs = self._compute_chunk_log_probs_loss_region(
                tea_logits_b, tea_ids, tea_mask, tea_chunks
            )
            # Student: only extract chunk positions from full logits (memory-efficient)
            stu_chunk_log_probs = self._compute_chunk_log_probs_full_seq(
                student_logits[b], stu_ids, stu_chunks
            )
            
            if len(tea_chunk_log_probs) == 0 or len(stu_chunk_log_probs) == 0:
                continue
            
            assert tea_chunk_log_probs.shape == stu_chunk_log_probs.shape, \
                f"Chunk log-prob shape mismatch: teacher {tea_chunk_log_probs.shape} vs student {stu_chunk_log_probs.shape}"
            
            # Compute binarised f-divergence
            chunk_losses = self._binarised_f_divergence(
                tea_chunk_log_probs.detach(),  # detach teacher
                stu_chunk_log_probs,
            )
            
            total_loss = total_loss + chunk_losses.sum()
            total_chunks += len(chunks)
            total_possible_chunks += max(stu_mask.sum().item(), tea_mask.sum().item())
        
        # Normalize by average token number (consistent with other algorithms)
        if total_chunks > 0:
            kd_loss = total_loss / avg_token_num
        else:
            kd_loss = total_loss
        
        align_ratio = torch.tensor(
            total_chunks / max(total_possible_chunks, 1),
            device=student_logits.device,
        )
        
        loss_info = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "align_ratio": align_ratio,
        }
        
        # Optionally add CE loss if kd_ratio < 1
        if self.args.kd.kd_ratio < 1:
            stu_logits_flat = student_logits[student_loss_mask]
            student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
            ce_loss = compute_cross_entropy(stu_logits_flat, student_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss
        
        return loss_info
