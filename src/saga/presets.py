"""Named scheduler configurations used in the e2e comparison.

Each preset bundles a ``ClusterConfig`` (cache policy + TTL behaviour) and a
``CoordinatorConfig`` (routing strategy, work stealing, fairness) so the same
simulator engine can stand in for any of the systems in the comparison table.
Numbers are aligned to each system's published behavior:

  * ``vllm``                 --- vLLM v0.6.0, V1 engine, LRU + FCFS
  * ``vllm_apc``             --- vLLM v0.15.1 with Automatic Prefix Caching
                                 and PrefixCacheAffinityRouter
  * ``sglang``               --- SGLang v0.5.8 with RadixAttention-like prefix
                                 sharing + cache-aware load balancing
  * ``llumnix``              --- vLLM + live migration (Llumnix v1.2)
  * ``trt_llm_scaffolding``  --- TensorRT-LLM v1.1 + Scaffolding multi-step
  * ``vllm_kvflow``          --- vLLM + KVFlow workflow-aware eviction
  * ``saga``                 --- SAGA (this paper): WA-LRU + tool-aware TTL
                                 + session affinity + AFS + work stealing
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from saga.scheduler.coordinator import CoordinatorConfig
from saga.sim.cluster import ClusterConfig


@dataclass(frozen=True)
class Preset:
    """A named (cluster, coordinator) bundle."""

    label: str
    description: str
    cluster: ClusterConfig
    coordinator: CoordinatorConfig


def _default_cluster(**overrides: object) -> ClusterConfig:
    cfg = ClusterConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _default_coord(**overrides: object) -> CoordinatorConfig:
    cfg = CoordinatorConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def preset_vllm() -> Preset:
    cluster = _default_cluster(eviction_policy="lru", ttl_enabled=False)
    coord = _default_coord(
        routing_strategy="least_loaded",
        enable_work_stealing=False,
        enable_afs=False,
    )
    return Preset("vllm", "vLLM v0.6.0 (V1 engine), LRU + FCFS", cluster, coord)


def preset_vllm_apc() -> Preset:
    cluster = _default_cluster(eviction_policy="lru_prefix", ttl_enabled=False)
    coord = _default_coord(
        routing_strategy="prefix_affinity",
        enable_work_stealing=False,
        enable_afs=False,
    )
    return Preset(
        "vllm_apc",
        "vLLM v0.15.1 with Automatic Prefix Caching + PrefixCacheAffinityRouter",
        cluster,
        coord,
    )


def preset_sglang() -> Preset:
    cluster = _default_cluster(eviction_policy="lru_prefix", ttl_enabled=False)
    coord = _default_coord(
        routing_strategy="prefix_affinity",
        enable_work_stealing=False,
        enable_afs=False,
        load_threshold=0.85,
    )
    return Preset(
        "sglang",
        "SGLang v0.5.8 with RadixAttention + cache-aware load balancing",
        cluster,
        coord,
    )


def preset_llumnix() -> Preset:
    cluster = _default_cluster(eviction_policy="lru", ttl_enabled=False)
    coord = _default_coord(
        routing_strategy="least_loaded",
        enable_work_stealing=True,
        enable_afs=False,
        t_idle_ms=100.0,
        r_max=2.0,
    )
    return Preset(
        "llumnix",
        "vLLM + live KV-cache migration (Llumnix v1.2)",
        cluster,
        coord,
    )


def preset_trt_llm_scaffolding() -> Preset:
    cluster = _default_cluster(eviction_policy="lru_prefix", ttl_enabled=False)
    coord = _default_coord(
        routing_strategy="session_affinity",
        enable_work_stealing=False,
        enable_afs=False,
        load_threshold=0.85,
    )
    return Preset(
        "trt_llm_scaffolding",
        "TensorRT-LLM v1.1 + Scaffolding (multi-step reasoning)",
        cluster,
        coord,
    )


def preset_vllm_kvflow() -> Preset:
    cluster = _default_cluster(eviction_policy="walru", ttl_enabled=True)
    coord = _default_coord(
        routing_strategy="session_affinity",
        enable_work_stealing=False,
        enable_afs=False,
    )
    return Preset(
        "vllm_kvflow",
        "vLLM + KVFlow workflow-aware eviction",
        cluster,
        coord,
    )


def preset_saga() -> Preset:
    cluster = _default_cluster(eviction_policy="walru", ttl_enabled=True)
    coord = _default_coord(
        routing_strategy="session_affinity",
        enable_work_stealing=True,
        enable_afs=True,
    )
    return Preset(
        "saga",
        "SAGA (this paper): WA-LRU + tool-aware TTL + AFS + work stealing",
        cluster,
        coord,
    )


# --------------------------------------------------------- ablations


def preset_saga_no_walru() -> Preset:
    p = preset_saga()
    cluster = _default_cluster(
        eviction_policy="lru",
        ttl_enabled=True,
    )
    return Preset("saga_no_walru", "SAGA w/o workflow-aware eviction", cluster, p.coordinator)


def preset_saga_no_ttl() -> Preset:
    p = preset_saga()
    cluster = _default_cluster(
        eviction_policy="walru",
        ttl_enabled=False,
    )
    return Preset("saga_no_ttl", "SAGA w/o tool-call-aware TTL", cluster, p.coordinator)


def preset_saga_no_prefetch() -> Preset:
    p = preset_saga()
    return Preset("saga_no_prefetch", "SAGA w/o speculative prefetch", p.cluster, p.coordinator)


def preset_saga_no_affinity() -> Preset:
    p = preset_saga()
    coord = _default_coord(
        routing_strategy="least_loaded",
        enable_work_stealing=True,
        enable_afs=True,
    )
    return Preset("saga_no_affinity", "SAGA w/o session affinity", p.cluster, coord)


def preset_saga_no_stealing() -> Preset:
    p = preset_saga()
    coord = _default_coord(
        routing_strategy="session_affinity",
        enable_work_stealing=False,
        enable_afs=True,
    )
    return Preset("saga_no_stealing", "SAGA w/o work stealing", p.cluster, coord)


def preset_saga_no_afs() -> Preset:
    p = preset_saga()
    coord = _default_coord(
        routing_strategy="session_affinity",
        enable_work_stealing=True,
        enable_afs=False,
    )
    return Preset("saga_no_afs", "SAGA w/o AFS fairness", p.cluster, coord)


# ------------------------------------------------------ registry


_PRESETS: dict[str, Callable[[], Preset]] = {
    "vllm": preset_vllm,
    "vllm_apc": preset_vllm_apc,
    "sglang": preset_sglang,
    "llumnix": preset_llumnix,
    "trt_llm_scaffolding": preset_trt_llm_scaffolding,
    "vllm_kvflow": preset_vllm_kvflow,
    "saga": preset_saga,
    "saga_no_walru": preset_saga_no_walru,
    "saga_no_ttl": preset_saga_no_ttl,
    "saga_no_prefetch": preset_saga_no_prefetch,
    "saga_no_affinity": preset_saga_no_affinity,
    "saga_no_stealing": preset_saga_no_stealing,
    "saga_no_afs": preset_saga_no_afs,
}


def get_preset(name: str) -> Preset:
    name = name.lower()
    fn = _PRESETS.get(name)
    if fn is None:
        raise ValueError(f"unknown preset {name!r}; valid: {sorted(_PRESETS)}")
    return fn()


def list_presets() -> list[str]:
    return sorted(_PRESETS.keys())


__all__ = [
    "Preset",
    "get_preset",
    "list_presets",
    "preset_llumnix",
    "preset_saga",
    "preset_saga_no_affinity",
    "preset_saga_no_afs",
    "preset_saga_no_prefetch",
    "preset_saga_no_stealing",
    "preset_saga_no_ttl",
    "preset_saga_no_walru",
    "preset_sglang",
    "preset_trt_llm_scaffolding",
    "preset_vllm",
    "preset_vllm_apc",
    "preset_vllm_kvflow",
]
