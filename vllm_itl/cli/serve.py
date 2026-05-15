"""Launch vLLM 0.15.1 with TOKEN_ITL enabled."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from vllm_itl.vllm.compat import install_patch


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _make_parser()
    args, vllm_args = parser.parse_known_args(argv)

    draft_model = args.token_itl_draft_model or os.getenv("VLLM_ITL_DRAFT_MODEL")
    if not draft_model:
        parser.error(
            "--token-itl-draft-model is required unless VLLM_ITL_DRAFT_MODEL is set."
        )

    _set_env_from_args(args, draft_model)
    vllm_args = _rewrite_or_add_speculative_config(
        vllm_args,
        num_speculative_tokens=args.token_itl_num_speculative_tokens,
    )
    install_patch()

    if vllm_args and vllm_args[0] == "serve":
        sys.argv = ["vllm", *vllm_args]
    else:
        sys.argv = ["vllm", "serve", *vllm_args]

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch vLLM with TokenTiming/TOKEN_ITL speculative decoding. "
            "All unrecognized arguments are forwarded to `vllm serve`."
        )
    )
    parser.add_argument("--token-itl-draft-model", default=None)
    parser.add_argument("--token-itl-num-speculative-tokens", type=int, default=5)
    parser.add_argument("--token-itl-max-draft-tokens", type=int, default=None)
    parser.add_argument("--token-itl-max-context-tokens", type=int, default=None)
    parser.add_argument("--token-itl-draft-device", default=None)
    parser.add_argument("--token-itl-draft-device-map", default=None)
    parser.add_argument("--token-itl-draft-dtype", default=None)
    parser.add_argument("--token-itl-dtw-window", type=int, default=None)
    parser.add_argument("--token-itl-max-cached-requests", type=int, default=None)
    parser.add_argument(
        "--token-itl-add-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--token-itl-draft-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--token-itl-allow-sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--token-itl-log-proposals",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _set_env_from_args(args: argparse.Namespace, draft_model: str) -> None:
    os.environ["VLLM_ITL_ENABLE"] = "1"
    os.environ["VLLM_ITL_DRAFT_MODEL"] = draft_model
    _set_env_if_not_none("VLLM_ITL_MAX_DRAFT_TOKENS", args.token_itl_max_draft_tokens)
    _set_env_if_not_none(
        "VLLM_ITL_MAX_CONTEXT_TOKENS",
        args.token_itl_max_context_tokens,
    )
    _set_env_if_not_none("VLLM_ITL_DRAFT_DEVICE", args.token_itl_draft_device)
    _set_env_if_not_none(
        "VLLM_ITL_DRAFT_DEVICE_MAP",
        args.token_itl_draft_device_map,
    )
    _set_env_if_not_none("VLLM_ITL_DRAFT_DTYPE", args.token_itl_draft_dtype)
    _set_env_if_not_none("VLLM_ITL_DTW_WINDOW", args.token_itl_dtw_window)
    _set_env_if_not_none(
        "VLLM_ITL_MAX_CACHED_REQUESTS",
        args.token_itl_max_cached_requests,
    )
    _set_env_bool_if_not_none(
        "VLLM_ITL_ADD_SPECIAL_TOKENS",
        args.token_itl_add_special_tokens,
    )
    _set_env_bool_if_not_none(
        "VLLM_ITL_ENABLE_DRAFT_CACHE",
        args.token_itl_draft_cache,
    )
    _set_env_bool_if_not_none(
        "VLLM_ITL_ALLOW_SAMPLING",
        args.token_itl_allow_sampling,
    )
    _set_env_bool_if_not_none(
        "VLLM_ITL_LOG_PROPOSALS",
        args.token_itl_log_proposals,
    )
    _enable_vllm_plugin()


def _rewrite_or_add_speculative_config(
    argv: list[str],
    *,
    num_speculative_tokens: int,
) -> list[str]:
    config = {
        "method": "ngram",
        "model": "ngram",
        "num_speculative_tokens": int(num_speculative_tokens),
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 1,
    }
    rewritten = list(argv)
    for index, item in enumerate(rewritten):
        if item == "--speculative-config" and index + 1 < len(rewritten):
            user_config = _parse_json_object(rewritten[index + 1])
            user_config.update(config)
            rewritten[index + 1] = json.dumps(user_config, separators=(",", ":"))
            return rewritten
        if item.startswith("--speculative-config="):
            user_config = _parse_json_object(item.split("=", 1)[1])
            user_config.update(config)
            rewritten[index] = (
                "--speculative-config="
                + json.dumps(user_config, separators=(",", ":"))
            )
            return rewritten

    rewritten.extend(
        [
            "--speculative-config",
            json.dumps(config, separators=(",", ":")),
        ]
    )
    return rewritten


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--speculative-config must be a JSON object.")
    return parsed


def _set_env_if_not_none(name: str, value: object | None) -> None:
    if value is not None:
        os.environ[name] = str(value)


def _set_env_bool_if_not_none(name: str, value: bool | None) -> None:
    if value is not None:
        os.environ[name] = "1" if value else "0"


def _enable_vllm_plugin() -> None:
    name = "vllm_itl"
    current = os.environ.get("VLLM_PLUGINS")
    if current is None:
        os.environ["VLLM_PLUGINS"] = name
        return
    plugins = [item for item in current.split(",") if item]
    if name not in plugins:
        plugins.append(name)
    os.environ["VLLM_PLUGINS"] = ",".join(plugins)


if __name__ == "__main__":
    main()
