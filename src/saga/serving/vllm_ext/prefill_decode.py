"""Separate-stream KV-cache prefetch binder.

The paper's speculative-prefetch optimisation (§4.3) overlaps the prefill of
the most-likely successor node with the running batch's decode. On vLLM v0.6
the natural seam is the ``model_executor`` -- specifically the prefill
launcher, which controls the CUDA stream used for the KV-block memcpy.

This binder

1.  Owns one *prefetch* CUDA stream per worker, separate from the *compute*
    stream used by the running batch. Decoupling the streams lets the GPU
    scheduler run them concurrently as long as their data dependencies do
    not cross.
2.  When SAGA's :class:`GlobalCoordinator` signals an upcoming successor,
    enqueues a :func:`saga.serving.cuda.prefetch_blocks` on the prefetch
    stream targeting the predicted block IDs.
3.  Records a CUDA event on the prefetch stream and inserts a wait edge on
    the compute stream the moment the successor enters the running batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from saga.utils.logging import get_logger


log = get_logger("saga.serving.vllm_ext.prefill_decode")


@dataclass
class PrefillDecodeBinder:
    """Bind SAGA's prefetch path to vLLM's prefill/decode executor.

    The binder is GPU-aware: it grabs handles to the worker's compute and
    prefetch CUDA streams when :meth:`install` is called, and falls back to
    no-op CPU-side prefetch in CI / simulator paths.
    """

    n_streams: int = 1
    _installed_on: Any = field(default=None, repr=False)
    _compute_stream: Any = field(default=None, repr=False)
    _prefetch_stream: Any = field(default=None, repr=False)
    _native_module: Any = field(default=None, repr=False)
    _n_prefetched: int = field(default=0, init=False, repr=False)

    # ----------------------------------------------------- install

    def install(self, vllm_executor: Any) -> None:
        if self._installed_on is not None:
            return
        try:
            import torch

            current = torch.cuda.current_stream()
            self._compute_stream = current
            self._prefetch_stream = torch.cuda.Stream()
        except (ImportError, RuntimeError, AssertionError):
            log.warning(
                "torch.cuda unavailable; prefetch binder runs in dry-run mode "
                "(prefetch_blocks() calls are no-ops)."
            )

        try:
            from saga import _cuda as native

            self._native_module = native
        except Exception:
            log.info("saga._cuda not available; using CPU fallback for prefetch.")

        self._installed_on = vllm_executor
        log.info(
            "PrefillDecodeBinder installed (compute_stream=%s, prefetch_stream=%s, "
            "native=%s)",
            self._compute_stream is not None,
            self._prefetch_stream is not None,
            self._native_module is not None,
        )

    def uninstall(self) -> None:
        self._installed_on = None
        self._compute_stream = None
        self._prefetch_stream = None
        self._native_module = None

    # ----------------------------------------------------- prefetch

    def prefetch_blocks(
        self,
        src_k: Any,
        src_v: Any,
        dst_k: Any,
        dst_v: Any,
        src_ids: Any,
        dst_ids: Any,
        shape: Any,
    ) -> int:
        """Launch a separate-stream prefetch; returns approximate bytes copied."""
        if self._native_module is None or self._prefetch_stream is None:
            self._n_prefetched += int(getattr(src_ids, "shape", [0])[0]) if hasattr(src_ids, "shape") else 0
            return 0

        stream_ptr = int(self._prefetch_stream.cuda_stream)
        bytes_copied = self._native_module.prefetch_blocks(
            src_k, src_v, dst_k, dst_v, src_ids, dst_ids, shape,
            stream_ptr=stream_ptr,
        )
        self._n_prefetched += int(getattr(src_ids, "shape", [0])[0]) if hasattr(src_ids, "shape") else 0
        return int(bytes_copied)

    def sync_compute_with_prefetch(self) -> None:
        """Insert a wait-edge so the compute stream sees the prefetched data.

        Called by the V1 engine hook the moment a prefetched session enters
        the running batch. No-op without ``torch.cuda``.
        """
        if self._compute_stream is None or self._prefetch_stream is None:
            return
        try:
            import torch

            event = torch.cuda.Event()
            event.record(self._prefetch_stream)
            event.wait(self._compute_stream)
        except (ImportError, RuntimeError, AssertionError):
            pass

    @property
    def stats(self) -> dict[str, int]:
        return {"n_prefetched_blocks": self._n_prefetched}
