"""TokenTiming / TOKEN_ITL integration for vLLM."""

from __future__ import annotations

from .alignment import AlignmentResult, AlignmentStep, dynamic_token_warping
from .config import TokenITLConfig, TokenTimingConfig
from .prob_mapping import (
    ProposalProbability,
    acceptance_probability,
    map_top1_draft_probabilities,
    selected_token_probabilities_from_logits,
)

__version__ = "0.1.1"
SUPPORTED_VLLM_VERSION = "0.15.1"
TOKEN_ITL_ALGORITHM = "TOKEN_ITL"

__all__ = [
    "AlignmentResult",
    "AlignmentStep",
    "ProposalProbability",
    "SUPPORTED_VLLM_VERSION",
    "TOKEN_ITL_ALGORITHM",
    "TokenITLConfig",
    "TokenTimingConfig",
    "__version__",
    "acceptance_probability",
    "dynamic_token_warping",
    "map_top1_draft_probabilities",
    "selected_token_probabilities_from_logits",
]
