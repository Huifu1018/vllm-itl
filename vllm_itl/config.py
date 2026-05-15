"""Configuration for TokenTiming and the vLLM TOKEN_ITL plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"0", "false", "off", "none"}:
        return None
    parsed = float(value)
    if parsed == 0:
        return None
    if parsed < 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


@dataclass(frozen=True)
class TokenTimingConfig:
    """Runtime knobs for standalone TokenTiming greedy decoding."""

    max_new_tokens: int = 128
    num_draft_tokens: int = 8
    dtw_window: int | None = 8
    temperature: float = 1.0
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    device: str | None = None
    target_device: str | None = None
    draft_device: str | None = None
    use_cache: bool = True
    add_special_tokens: bool = False
    max_proxy_tokens_per_step: int | None = None

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.num_draft_tokens <= 0:
            raise ValueError("num_draft_tokens must be positive.")
        if self.dtw_window is not None and self.dtw_window < 0:
            raise ValueError("dtw_window must be non-negative or None.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if (
            self.max_proxy_tokens_per_step is not None
            and self.max_proxy_tokens_per_step <= 0
        ):
            raise ValueError("max_proxy_tokens_per_step must be positive or None.")

    @property
    def effective_target_device(self) -> str | None:
        return self.target_device or self.device

    @property
    def effective_draft_device(self) -> str | None:
        return self.draft_device or self.device


@dataclass(frozen=True)
class TokenITLConfig:
    """Runtime knobs read by the vLLM TOKEN_ITL plugin."""

    draft_model: str | None = None
    draft_device: str | None = None
    draft_device_map: str | None = None
    draft_dtype: str = "auto"
    dtw_window: int | None = 8
    max_draft_tokens: int | None = None
    max_context_tokens: int | None = None
    max_cached_requests: int = 256
    add_special_tokens: bool = False
    enable_draft_cache: bool = True
    clone_draft_cache: bool = True
    allow_sampling: bool = True
    log_proposals: bool = False
    metrics_log_interval: float | None = 60.0

    @classmethod
    def from_env(cls) -> "TokenITLConfig":
        return cls(
            draft_model=os.getenv("VLLM_ITL_DRAFT_MODEL") or None,
            draft_device=os.getenv("VLLM_ITL_DRAFT_DEVICE") or None,
            draft_device_map=os.getenv("VLLM_ITL_DRAFT_DEVICE_MAP") or None,
            draft_dtype=os.getenv("VLLM_ITL_DRAFT_DTYPE", "auto"),
            dtw_window=_env_int("VLLM_ITL_DTW_WINDOW", 8),
            max_draft_tokens=_env_int("VLLM_ITL_MAX_DRAFT_TOKENS", None),
            max_context_tokens=_env_int("VLLM_ITL_MAX_CONTEXT_TOKENS", None),
            max_cached_requests=_env_int("VLLM_ITL_MAX_CACHED_REQUESTS", 256) or 256,
            add_special_tokens=_env_bool("VLLM_ITL_ADD_SPECIAL_TOKENS", False),
            enable_draft_cache=_env_bool("VLLM_ITL_ENABLE_DRAFT_CACHE", True),
            clone_draft_cache=_env_bool("VLLM_ITL_CLONE_DRAFT_CACHE", True),
            allow_sampling=_env_bool("VLLM_ITL_ALLOW_SAMPLING", True),
            log_proposals=_env_bool("VLLM_ITL_LOG_PROPOSALS", False),
            metrics_log_interval=_env_float("VLLM_ITL_METRICS_LOG_INTERVAL", 60.0),
        )
