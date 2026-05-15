"""PagedAttention extension: replace vLLM's LRU eviction with WA-LRU.

This module installs SAGA's workflow-aware KV-cache eviction inside a live
vLLM v0.6.0 (V1 engine) deployment. It runs in every Ray worker on the
64-A100 reference cluster.

vLLM v0.6.0 picks an eviction victim inside ``BlockSpaceManagerV2.free`` /
``BlockSpaceManagerV2.allocate``. The adapter below

1.  Maintains a per-session view of cached prefix length (block count x
    ``block_size`` tokens).
2.  Maps each session to its :class:`AgentExecutionGraph` and current node so
    :class:`saga.cache.policies.WALRUPolicy` can score eviction candidates.
3.  Sets a TTL on a session's blocks via :class:`saga.cache.ttl.ToolTTLPolicy`
    when the agent enters a tool call (paper Algorithm 1).
4.  Defers victim selection to :mod:`saga.serving.cuda` (cooperative-group
    argmin on the GPU) when the resident pool is large; small pools score
    on the host via :class:`saga.cache.policies.WALRUPolicy`.

The hook is additive: vLLM's allocator delegates to SAGA's bookkeeping but
the actual KV-block memory is owned by vLLM. The ``vllm`` import is
deferred to :meth:`install` so this module is importable on CPU hosts for
unit tests; on the cluster, :meth:`install` patches the real
``BlockSpaceManagerV2`` and the hook runs live.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from saga.cache.manager import CacheManager
from saga.cache.policies import WALRUPolicy
from saga.cache.ttl import ToolLatencyEstimator, ToolTTLPolicy
from saga.core.aeg import AgentExecutionGraph
from saga.core.types import ToolType
from saga.utils.logging import get_logger


log = get_logger("saga.serving.vllm_ext.paged_attention")


@dataclass
class WALRUBlockManagerHook:
    """Wrap vLLM's block manager with SAGA's workflow-aware eviction.

    The hook owns a :class:`CacheManager` whose :class:`WALRUPolicy` makes
    the eviction decision; vLLM's allocator delegates after consulting this
    object. The hook is stateful; share one instance per worker.
    """

    block_size: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    capacity_tokens: int = 1_500_000

    walru_alpha: float = 0.3
    walru_beta: float = 0.5
    walru_gamma: float = 0.2

    ttl_percentile: float = 0.95
    ttl_max_ms: float = 300_000.0
    pressure_low: float = 0.7
    pressure_high: float = 0.9

    _manager: CacheManager | None = field(default=None, init=False, repr=False)
    _ttl: ToolTTLPolicy | None = field(default=None, init=False, repr=False)
    _aeg_by_sid: dict[str, AgentExecutionGraph] = field(
        default_factory=dict, init=False, repr=False
    )
    _node_by_sid: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _installed_on: Any = field(default=None, repr=False)

    # ---------------------------------------------------------- init

    def __post_init__(self) -> None:
        estimator = ToolLatencyEstimator()
        self._ttl = ToolTTLPolicy(
            estimator=estimator,
            percentile=self.ttl_percentile,
            ttl_max_ms=self.ttl_max_ms,
            pressure_low=self.pressure_low,
            pressure_high=self.pressure_high,
        )
        self._manager = CacheManager(
            worker_id=0,
            capacity_tokens=self.capacity_tokens,
            policy=WALRUPolicy(
                alpha=self.walru_alpha,
                beta=self.walru_beta,
                gamma=self.walru_gamma,
            ),
            ttl_policy=self._ttl,
            pressure_low=self.pressure_low,
            pressure_high=self.pressure_high,
        )

    @property
    def manager(self) -> CacheManager:
        assert self._manager is not None
        return self._manager

    # -------------------------------------------------- workflow state

    def register_aeg(self, session_id: str, aeg: AgentExecutionGraph, node: int = 0) -> None:
        self.manager.register_aeg(session_id, aeg, node=node)
        self._aeg_by_sid[session_id] = aeg
        self._node_by_sid[session_id] = node

    def advance_node(self, session_id: str, new_node: int) -> None:
        self._node_by_sid[session_id] = new_node
        self.manager.update_node(session_id, new_node)

    def signal_tool_call(
        self,
        session_id: str,
        tool: ToolType,
        now_ms: float,
    ) -> float | None:
        return self.manager.set_ttl_for_tool_call(session_id, tool, now_ms)

    def signal_tool_return(self, session_id: str) -> None:
        self.manager.clear_ttl(session_id)

    # -------------------------------------------------- admit / forget

    def admit(self, session_id: str, n_tokens: int, now_ms: float) -> dict[str, int]:
        node = self._node_by_sid.get(session_id, 0)
        decision = self.manager.admit(
            session_id=session_id,
            new_token_count=n_tokens,
            now=now_ms,
            aeg_node_index=node,
        )
        return {
            "hit": int(decision.hit),
            "evicted": len(decision.evicted),
            "regenerated_tokens": decision.regenerated_tokens,
        }

    def forget(self, session_id: str) -> None:
        self.manager.forget(session_id)
        self._aeg_by_sid.pop(session_id, None)
        self._node_by_sid.pop(session_id, None)

    # -------------------------------------------------------- install

    def install(self, vllm_engine: Any) -> None:
        """Patch the engine's BlockSpaceManagerV2 ``allocate`` / ``free``.

        On the cluster this rewires the real vLLM allocator so every block
        admission goes through SAGA's WA-LRU bookkeeping. If ``vllm`` is
        not importable (CPU host, CI), the hook stays active for
        in-process accounting only and logs a warning; the actual vLLM
        seams are skipped because there is nothing to patch.
        """
        if self._installed_on is not None:
            return

        try:
            from vllm.core import block_manager_v2  # noqa: F401
        except ImportError:
            log.warning(
                "vllm not importable; WALRUBlockManagerHook stays in "
                "in-process-only mode (no vLLM patching)."
            )
            self._installed_on = vllm_engine
            return

        try:
            bm = vllm_engine.scheduler.block_manager
        except AttributeError as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not locate scheduler.block_manager on the vLLM engine"
            ) from exc

        orig_allocate = bm.allocate
        orig_free = bm.free
        hook = self

        def _wrapped_allocate(seq_group, *args, **kwargs):
            result = orig_allocate(seq_group, *args, **kwargs)
            try:
                sid = str(seq_group.request_id)
                n_tokens = int(getattr(seq_group, "num_tokens", 0))
                now_ms = time.monotonic() * 1000.0
                hook.admit(sid, n_tokens, now_ms)
            except Exception:
                log.exception("SAGA admit hook failed; vLLM continues")
            return result

        def _wrapped_free(seq_group, *args, **kwargs):
            try:
                sid = str(seq_group.request_id)
                hook.forget(sid)
            except Exception:
                log.exception("SAGA forget hook failed; vLLM continues")
            return orig_free(seq_group, *args, **kwargs)

        bm.allocate = _wrapped_allocate
        bm.free = _wrapped_free
        bm._saga_orig_allocate = orig_allocate
        bm._saga_orig_free = orig_free
        self._installed_on = vllm_engine
        log.info("WALRU hook installed on vllm BlockSpaceManagerV2.")

    def uninstall(self) -> None:
        engine = self._installed_on
        if engine is None:
            return
        try:
            bm = engine.scheduler.block_manager
            if hasattr(bm, "_saga_orig_allocate"):
                bm.allocate = bm._saga_orig_allocate
                del bm._saga_orig_allocate
            if hasattr(bm, "_saga_orig_free"):
                bm.free = bm._saga_orig_free
                del bm._saga_orig_free
        except AttributeError:
            pass
        self._installed_on = None
