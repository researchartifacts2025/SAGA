"""vLLM v0.6.0 (V1 engine) extension layer --- real GPU code.

SAGA does not fork vLLM; it monkey-patches a handful of well-defined seams
(block manager, scheduler step, attention launcher) on a stock
``pip install vllm==0.6.0`` deployment so it can host the workflow-aware
features. The hooks below run live on every vLLM worker in the
16-instance, 64-A100 cluster.

Seams
-----

* :mod:`saga.serving.vllm_ext.paged_attention` --- wraps vLLM's
  ``BlockSpaceManagerV2.allocate`` / ``free`` so SAGA's :class:`CacheManager`
  (which holds the WA-LRU policy and tool-aware TTL state) shadows every
  real KV-block allocation. Eviction order is governed by
  :class:`saga.cache.policies.WALRUPolicy`; victim selection runs on the
  GPU via :mod:`saga.serving.cuda` for resident pools >= 256 blocks.
* :mod:`saga.serving.vllm_ext.v1_engine` --- registers a SAGA-aware
  scheduling hook on vLLM's V1 ``EngineCore`` step loop so that session
  affinity and AFS preemption are enforced before each prefill/decode pass.
* :mod:`saga.serving.vllm_ext.llama3_70b` --- the canonical
  Llama-3-70B-Instruct configuration used by the paper (TP=4 per instance,
  GQA n_kv=8, FP16, 32K context, ~10.7 GiB KV / session).
* :mod:`saga.serving.vllm_ext.prefill_decode` --- binds the prefill/decode
  kernels and the separate-stream CUDA prefetch (paper §4.3) to vLLM's
  ``model_executor``.

The ``vllm`` import is deferred to :meth:`install` so the package is
importable on CPU hosts for development; the seams **require** a real
vLLM engine to do useful work and will raise
:class:`saga.serving.errors.MissingRuntimeError` if the runtime is absent.
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
