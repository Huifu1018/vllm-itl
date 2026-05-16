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


@dataclass(frozen=True)
class _TPRuntime:
    rank: int
    world_size: int
    group: Any | None


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

        self.tp_runtime = _detect_tp_runtime()
        if self.config.draft_tp_rank >= self.tp_runtime.world_size:
            raise ValueError(
                "VLLM_ITL_DRAFT_TP_RANK must be smaller than tensor parallel "
                f"size ({self.config.draft_tp_rank} >= {self.tp_runtime.world_size})."
            )
        if (
            self.tp_runtime.world_size > 1
            and self.config.draft_tp_rank != 0
            and getattr(self.tp_runtime.group, "mq_broadcaster", None) is not None
        ):
            raise ValueError(
                "This vLLM TP runtime supports object broadcast only from TP "
                "rank 0; set VLLM_ITL_DRAFT_TP_RANK=0."
            )
        self._runs_draft_model = self.tp_runtime.rank == self.config.draft_tp_rank
        self.target_runtime = self._build_target_runtime(vllm_config)
        self.target_tokenizer = (
            self._load_target_tokenizer() if self._runs_draft_model else None
        )
        self._hf_proposer: HFDraftProposer | None = None
        if self._runs_draft_model:
            logger.info(
                "TOKEN_ITL TP rank %s/%s will run draft model %s.",
                self.tp_runtime.rank,
                self.tp_runtime.world_size,
                self.config.draft_model,
            )
        elif self.tp_runtime.world_size > 1:
            logger.info(
                "TOKEN_ITL TP rank %s/%s will receive draft proposals from "
                "TP rank %s; local draft model load is skipped.",
                self.tp_runtime.rank,
                self.tp_runtime.world_size,
                self.config.draft_tp_rank,
            )

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        if self._runs_draft_model:
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
        req_ids = list(req_ids or [f"row:{i}" for i in range(len(sampled_token_ids))])
        if self._runs_draft_model:
            try:
                draft_token_ids = self._propose_local(
                    sampled_token_ids,
                    num_tokens_no_spec,
                    token_ids_cpu,
                    sampling_metadata=sampling_metadata,
                    req_ids=req_ids,
                )
                payload = self._export_payload(draft_token_ids)
            except Exception as exc:
                if self.tp_runtime.world_size > 1:
                    self._broadcast_payload(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "draft_token_ids": [],
                        }
                    )
                raise
        else:
            payload = None

        payload = self._broadcast_payload(payload)
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(
                "TOKEN_ITL proposal failed on draft TP rank "
                f"{self.config.draft_tp_rank}: {payload['error']}"
            )
        if self._runs_draft_model:
            return draft_token_ids
        return self._apply_payload(payload)

    def _propose_local(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        *,
        sampling_metadata: Any = None,
        req_ids: Sequence[str],
    ) -> list[list[int]]:
        proposer = self._ensure_hf_proposer()
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

            max_proxy_tokens = min(
                self.k,
                self.target_runtime.max_model_len - num_tokens,
            )
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

    def _export_payload(self, draft_token_ids: list[list[int]]) -> dict[str, Any]:
        return {
            "draft_token_ids": [
                [int(token_id) for token_id in row] for row in draft_token_ids
            ],
        }

    def _apply_payload(self, payload: Any) -> list[list[int]]:
        if not isinstance(payload, dict):
            raise RuntimeError("TOKEN_ITL TP broadcast did not return a proposal payload.")
        return [
            [int(token_id) for token_id in row]
            for row in payload.get("draft_token_ids", [])
        ]

    def _broadcast_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if self.tp_runtime.world_size == 1:
            assert payload is not None
            return payload
        if self.tp_runtime.group is None:
            raise RuntimeError(
                "TOKEN_ITL detected tensor parallelism but could not access "
                "the vLLM TP communication group."
            )
        return self.tp_runtime.group.broadcast_object(
            payload,
            src=self.config.draft_tp_rank,
        )

    def stats_snapshot(self) -> dict[str, int]:
        proposer = self._hf_proposer
        if proposer is None:
            return {}
        return proposer.stats.snapshot()

    def _ensure_hf_proposer(self) -> HFDraftProposer:
        if self.target_tokenizer is None:
            raise RuntimeError(
                "TOKEN_ITL draft proposer was requested on a TP rank that is "
                "not configured to run the draft model."
            )
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


def _detect_tp_runtime() -> _TPRuntime:
    try:
        from vllm.distributed import (  # type: ignore
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )
    except Exception:
        return _TPRuntime(rank=0, world_size=1, group=None)

    try:
        rank = int(get_tensor_model_parallel_rank())
        world_size = int(get_tensor_model_parallel_world_size())
        group = get_tp_group() if world_size > 1 else None
        return _TPRuntime(rank=rank, world_size=world_size, group=group)
    except Exception:
        logger.debug("Could not detect vLLM tensor-parallel runtime.", exc_info=True)
        return _TPRuntime(rank=0, world_size=1, group=None)
