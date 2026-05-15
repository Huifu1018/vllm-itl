"""vLLM-compatible TokenTiming proposer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from vllm_itl.config import TokenITLConfig
from vllm_itl.hf_proposer import HFDraftProposer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TargetRuntime:
    model_id: str
    tokenizer_id: str
    trust_remote_code: bool
    max_model_len: int


class VllmTokenITLProposer:
    """Drop-in replacement for vLLM 0.15.1 ``NgramProposer``."""

    is_token_itl_proposer = True

    def __init__(self, vllm_config: Any) -> None:
        assert vllm_config.speculative_config is not None
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.k = int(self.speculative_config.num_speculative_tokens)
        self.config = TokenITLConfig.from_env()
        if not self.config.draft_model:
            raise ValueError(
                "VLLM_ITL_DRAFT_MODEL must be set. Use "
                "vllm-itl-serve --token-itl-draft-model ..."
            )

        self.target_runtime = self._build_target_runtime(vllm_config)
        self.target_tokenizer = self._load_target_tokenizer()
        self._hf_proposer: HFDraftProposer | None = None

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_hf_proposer()

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        slot_mappings: Any = None,
        sampling_metadata: Any = None,
        req_ids: Sequence[str] | None = None,
    ) -> list[list[int]]:
        proposer = self._ensure_hf_proposer()
        req_ids = list(req_ids or [f"row:{i}" for i in range(len(sampled_token_ids))])
        draft_token_ids: list[list[int]] = []

        for index, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                draft_token_ids.append([])
                continue

            if not self.config.allow_sampling and not _is_request_greedy(
                sampling_metadata,
                index,
            ):
                draft_token_ids.append([])
                continue

            num_tokens = int(num_tokens_no_spec[index])
            if num_tokens >= self.target_runtime.max_model_len:
                draft_token_ids.append([])
                continue

            max_proxy_tokens = min(self.k, self.target_runtime.max_model_len - num_tokens)
            if max_proxy_tokens <= 0:
                draft_token_ids.append([])
                continue

            current_target_ids = _slice_row(token_ids_cpu, index, num_tokens)
            current_text = _decode_ids(self.target_tokenizer, current_target_ids)
            proposal = proposer.propose(
                str(req_ids[index]),
                current_text,
                max_proxy_tokens=max_proxy_tokens,
            )
            proxy_ids = list(proposal.proxy_target_token_ids[:max_proxy_tokens])
            if self.config.log_proposals:
                logger.info(
                    "TOKEN_ITL proposal req=%s proxy_tokens=%d draft_tokens=%d "
                    "cache=%s alignment_cost=%s",
                    req_ids[index],
                    len(proxy_ids),
                    len(proposal.draft_token_ids),
                    proposal.cache_event,
                    proposal.alignment_cost,
                )
            draft_token_ids.append(proxy_ids)

        return draft_token_ids

    def stats_snapshot(self) -> dict[str, int]:
        proposer = self._hf_proposer
        if proposer is None:
            return {}
        return proposer.stats.snapshot()

    def _ensure_hf_proposer(self) -> HFDraftProposer:
        if self._hf_proposer is None:
            logger.info(
                "Loading TOKEN_ITL draft model %s for target tokenizer %s.",
                self.config.draft_model,
                self.target_runtime.tokenizer_id,
            )
            self._hf_proposer = HFDraftProposer(
                draft_model_path=str(self.config.draft_model),
                target_tokenizer=self.target_tokenizer,
                config=self.config,
                trust_remote_code=self.target_runtime.trust_remote_code,
            )
        return self._hf_proposer

    def _load_target_tokenizer(self) -> object:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.target_runtime.tokenizer_id,
            trust_remote_code=self.target_runtime.trust_remote_code,
        )

    @staticmethod
    def _build_target_runtime(vllm_config: Any) -> _TargetRuntime:
        model_config = vllm_config.model_config
        model_id = str(getattr(model_config, "model"))
        tokenizer_id = str(getattr(model_config, "tokenizer", None) or model_id)
        trust_remote_code = bool(getattr(model_config, "trust_remote_code", False))
        max_model_len = int(getattr(model_config, "max_model_len"))
        return _TargetRuntime(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
        )


def _slice_row(token_ids_cpu: Any, index: int, length: int) -> tuple[int, ...]:
    row = token_ids_cpu[index, :length]
    tolist = getattr(row, "tolist", None)
    if callable(tolist):
        row = tolist()
    return tuple(int(token_id) for token_id in row)


def _decode_ids(tokenizer: object, token_ids: Sequence[int]) -> str:
    ids = [int(token_id) for token_id in token_ids]
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids)


def _is_request_greedy(sampling_metadata: Any, index: int) -> bool:
    if sampling_metadata is None:
        return True
    if bool(getattr(sampling_metadata, "all_greedy", False)):
        return True
    temperature = getattr(sampling_metadata, "temperature", None)
    if temperature is None:
        return False
    return float(_tensor_item(temperature, index, 1.0)) == 0.0


def _tensor_item(value: Any, index: int, default: float | int) -> float | int:
    if value is None:
        return default
    try:
        item = value[index]
    except Exception:
        item = value
    if hasattr(item, "item"):
        item = item.item()
    return item
