# vLLM 0.15.1 TOKEN_ITL Integration

This package targets vLLM 0.15.1 exactly.

vLLM's same-tokenizer `draft_model` mode cannot handle ordinary draft models
with heterogeneous tokenizers. TOKEN_ITL therefore uses vLLM's `ngram`
speculative slot as an engine hook, replacing the in-tree ngram proposer with a
TokenTiming proposer.

Runtime path:

1. Reconstruct the request's current target text from target token ids.
2. Encode the text with the draft tokenizer.
3. Reuse a conservative per-request HF draft KV cache when the draft-token
   context extends the cached prefix exactly.
4. Generate a short greedy draft block with the ordinary HF draft model.
5. Decode the draft block to text and retokenize it with the target tokenizer.
6. Optionally compute DTW alignment cost for diagnostics.
7. Return target-token proxy candidates to vLLM's speculative verifier.

vLLM performs the target verification, request mutation, scheduler accounting,
and speculative metrics. The default vLLM rejection sampler handles greedy and
sampling requests with deterministic-proposal semantics when no draft
probability rows are provided.

The integration is deliberately version-pinned because it patches internal
vLLM v1 runner methods.
