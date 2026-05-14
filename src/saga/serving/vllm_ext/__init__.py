"""vLLM v0.6.0 (V1 engine) extension layer.

SAGA does not fork vLLM; it monkey-patches a handful of well-defined seams
(block manager, scheduler step, attention launcher) so a stock
``pip install vllm==0.6.0`` deployment can host the workflow-aware features.
This package contains those seams.

Seams
-----

* :mod:`saga.serving.vllm_ext.paged_attention` --- wraps vLLM's
  ``BlockSpaceManagerV2.allocate`` / ``free`` so SAGA's :class:`CacheManager`
  (which holds the WA-LRU policy and tool-aware TTL state) shadows every
  block allocation. Eviction order is governed by
  :class:`saga.cache.policies.WALRUPolicy`.
* :mod:`saga.serving.vllm_ext.v1_engine` --- registers a SAGA-aware
  scheduling hook on vLLM's V1 ``EngineCore`` step loop so that session
  affinity and AFS preemption can be enforced before each forward pass.
* :mod:`saga.serving.vllm_ext.llama3_70b` --- the canonical Llama-3-70B
  configuration used by the paper (TP=4 per instance, GQA n_kv=8, FP16).
* :mod:`saga.serving.vllm_ext.prefill_decode` --- binds the prefill/decode
  kernels and the separate-stream prefetch hook to vLLM's executor.

Importing any of these modules is safe without vLLM installed: each module
imports ``vllm`` lazily inside :func:`install` so the rest of SAGA continues
to run in simulator mode.
"""

from __future__ import annotations

from saga.serving.vllm_ext.llama3_70b import (
    LLAMA3_8B,
    LLAMA3_70B,
    ModelConfig,
    assert_paper_invariants,
)
from saga.serving.vllm_ext.paged_attention import WALRUBlockManagerHook
from saga.serving.vllm_ext.prefill_decode import PrefillDecodeBinder
from saga.serving.vllm_ext.v1_engine import V1EngineHook, install


__all__ = [
    "LLAMA3_8B",
    "LLAMA3_70B",
    "ModelConfig",
    "PrefillDecodeBinder",
    "V1EngineHook",
    "WALRUBlockManagerHook",
    "assert_paper_invariants",
    "install",
]
