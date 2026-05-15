# vLLM ITL

`vllm-itl` integrates the second paper's **TokenTiming / TOKEN_ITL** method with
**vLLM 0.15.1**.

It is intended for heterogeneous-vocabulary speculative decoding when:

- the target model and draft model use different tokenizers,
- the draft model is an ordinary HF `AutoModelForCausalLM`,
- there is no dedicated MTP/EAGLE/P-EAGLE draft checkpoint.

## What It Implements

- Dynamic Token Warping (DTW) alignment utilities.
- HF draft-model candidate generation with conservative per-request KV cache.
- Retokenization of draft text into target-vocabulary proxy tokens.
- vLLM `vllm.general_plugins` integration.
- `vllm-itl-serve` wrapper that launches normal `vllm serve`.

The implementation uses vLLM's `ngram` speculative slot as the engine hook. It
does not use vLLM's same-tokenizer `draft_model` mode.

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install "vllm-itl[vllm] @ git+https://github.com/Huifu1018/vllm-itl.git"
vllm-itl-preflight
```

With pip:

```bash
python -m pip install "vllm-itl[vllm] @ git+https://github.com/Huifu1018/vllm-itl.git"
```

The package pins the vLLM extra to `vllm==0.15.1`.

## Start A Server

```bash
vllm-itl-serve nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --token-itl-draft-model Qwen/Qwen2.5-1.5B-Instruct \
  --token-itl-num-speculative-tokens 5 \
  --token-itl-dtw-window 8
```

All ordinary vLLM arguments are forwarded unchanged. The wrapper consumes only
`--token-itl-*` flags and injects:

```json
{
  "method": "ngram",
  "model": "ngram",
  "num_speculative_tokens": 5,
  "prompt_lookup_min": 1,
  "prompt_lookup_max": 1
}
```

The in-process plugin replaces the ngram proposer with TOKEN_ITL before vLLM
starts model workers.

## Useful Flags

- `--token-itl-draft-model`: ordinary HF draft model path.
- `--token-itl-num-speculative-tokens`: max proxy target tokens per step.
- `--token-itl-max-draft-tokens`: cap draft-side token generation while trying
  to collect enough proxy tokens.
- `--token-itl-max-context-tokens`: optional draft-side context truncation.
- `--token-itl-draft-device`: move the HF draft model to a device.
- `--token-itl-draft-device-map`: pass a Transformers `device_map`.
- `--token-itl-draft-dtype`: `auto`, `float16`, `bfloat16`, or `float32`.
- `--token-itl-dtw-window`: DTW window for alignment diagnostics.
- `--no-token-itl-draft-cache`: disable per-request draft KV cache.
- `--no-token-itl-allow-sampling`: disable speculation for non-greedy requests.
- `--token-itl-log-proposals`: log proxy length, draft length, cache event, and
  DTW alignment cost.

The same settings can be supplied through `VLLM_ITL_*` environment variables.

## Sampling Semantics

TOKEN_ITL generates deterministic proxy candidates from the current request
state. In vLLM, proposals are verified by the standard speculative sampler. For
greedy requests, this is target-greedy-equivalent. For non-greedy requests, the
default path uses vLLM's deterministic-proposal rejection behavior with no draft
probability rows.

If you only want the conservative greedy route, run:

```bash
vllm-itl-serve ... --no-token-itl-allow-sampling
```

## Metrics

vLLM exports speculative decoding metrics. Acceptance rate:

```promql
rate(vllm:spec_decode_num_accepted_tokens_total[1m])
/
rate(vllm:spec_decode_num_draft_tokens_total[1m])
```

Accepted tokens per step:

```promql
rate(vllm:spec_decode_num_accepted_tokens_total[1m])
/
rate(vllm:spec_decode_num_drafts[1m])
```

## Development Checks

```bash
python -m unittest discover -s tests
python -m compileall vllm_itl tests
python -m build
```

## Current Boundary

This is an engine-integrated vLLM package, not a standalone HF demo. It is still
version-pinned to vLLM 0.15.1 internals and should be validated on the actual
GPU serving host before production use.
