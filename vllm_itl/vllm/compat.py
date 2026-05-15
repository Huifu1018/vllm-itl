"""Compatibility patch for vLLM 0.15.1 speculative decoding."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def install_patch() -> None:
    """Install TOKEN_ITL into vLLM's ngram speculative slot."""

    global _PATCHED
    if _PATCHED:
        return

    from vllm_itl.vllm.proposer import VllmTokenITLProposer

    ngram_mod = importlib.import_module("vllm.v1.spec_decode.ngram_proposer")
    setattr(ngram_mod, "NgramProposer", VllmTokenITLProposer)

    runner_mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    setattr(runner_mod, "NgramProposer", VllmTokenITLProposer)
    _patch_gpu_model_runner(runner_mod)

    _PATCHED = True
    logger.info("Installed vLLM TOKEN_ITL patch for vLLM 0.15.1.")


def _patch_gpu_model_runner(runner_mod: ModuleType) -> None:
    cls = getattr(runner_mod, "GPUModelRunner")
    if not hasattr(cls, "_token_itl_original_propose_draft_token_ids"):
        cls._token_itl_original_propose_draft_token_ids = cls.propose_draft_token_ids
        cls.propose_draft_token_ids = _patched_propose_draft_token_ids


def _is_token_itl_drafter(drafter: object) -> bool:
    return bool(getattr(drafter, "is_token_itl_proposer", False))


def _patched_propose_draft_token_ids(
    self: Any,
    scheduler_output: Any,
    sampled_token_ids: Any,
    sampling_metadata: Any,
    hidden_states: Any,
    sample_hidden_states: Any,
    aux_hidden_states: Any,
    spec_decode_metadata: Any,
    common_attn_metadata: Any,
    slot_mappings: Any,
) -> Any:
    spec_config = self.speculative_config
    drafter = getattr(self, "drafter", None)
    if (
        spec_config is not None
        and spec_config.method == "ngram"
        and _is_token_itl_drafter(drafter)
    ):
        if not isinstance(sampled_token_ids, list):
            raise TypeError("TOKEN_ITL ngram path expects CPU list sampled_token_ids.")
        return drafter.propose(
            sampled_token_ids,
            self.input_batch.num_tokens_no_spec,
            self.input_batch.token_ids_cpu,
            slot_mappings=slot_mappings,
            sampling_metadata=sampling_metadata,
            req_ids=list(self.input_batch.req_ids),
        )

    return self._token_itl_original_propose_draft_token_ids(
        scheduler_output,
        sampled_token_ids,
        sampling_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        spec_decode_metadata,
        common_attn_metadata,
        slot_mappings,
    )
