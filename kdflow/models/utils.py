import logging
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


@dataclass
class TokenizerCompareResult:
    """Result of comparing two tokenizers."""
    template_identical: bool = True
    vocab_identical: bool = True

    @property
    def is_identical(self) -> bool:
        """True if both chat_template and vocabulary are identical."""
        return self.template_identical and self.vocab_identical


def check_tokenizer_identical(
    tokenizer1: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    tokenizer2: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> TokenizerCompareResult:
    """
    Check if two tokenizers are identical by comparing their chat_template and vocabulary.

    Args:
        tokenizer1: The first tokenizer to compare.
        tokenizer2: The second tokenizer to compare.

    Returns:
        TokenizerCompareResult with template_identical and vocab_identical fields.
    """
    result = TokenizerCompareResult()

    # Compare chat_template
    chat_template1 = getattr(tokenizer1, "chat_template", None)
    chat_template2 = getattr(tokenizer2, "chat_template", None)
    if chat_template1 != chat_template2:
        logger.warning("Tokenizers differ in chat_template.")
        result.template_identical = False

    # Compare vocabulary
    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    if vocab1 != vocab2:
        logger.warning(
            f"Tokenizers differ in vocabulary. "
            f"Vocab sizes: {len(vocab1)} vs {len(vocab2)}."
        )
        result.vocab_identical = False

    return result
