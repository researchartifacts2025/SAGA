"""Optional native-acceleration wrapper.

If the compiled ``saga._native`` extension is available (built via
``python setup_native.py build_ext --inplace`` or via the CMake target) the
hot WA-LRU and Belady kernels use it. Otherwise the pure-Python paths run
unchanged. There is no API split: every public function in this module is
defined regardless of whether the native module loaded.

Detect the active backend at runtime via :func:`is_native_available` and
:func:`build_info`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence


# --------------------------------------------------------------- detection

try:
    from saga import _native as _ext  # type: ignore[attr-defined]

    _HAS_NATIVE = True
except Exception:  # pragma: no cover - import error path
    _ext = None
    _HAS_NATIVE = False


def is_native_available() -> bool:
    """Return ``True`` iff the compiled ``saga._native`` module loaded."""
    return _HAS_NATIVE


def build_info() -> str:
    """One-line description of the active backend."""
    if _HAS_NATIVE and _ext is not None:
        try:
            return str(_ext.build_info())
        except Exception:
            return "saga native (no build_info)"
    return "saga pure-python fallback"


# --------------------------------------------------------------- WA-LRU


def walru_select_victim(
    entries: Sequence[CacheEntryProtocol],
    now: float,
    tau_max: float,
    size_max: int,
    alpha: float,
    beta: float,
    gamma: float,
    reuse_lookup: Callable[[str], float] | None = None,
) -> int:
    """Return the index of the lowest-score (i.e. most-evictable) entry.

    ``reuse_lookup`` is queried once per non-pinned entry to obtain the
    predicted reuse value; the lookup result is then passed into the kernel
    via a ``CacheEntryView`` so that the native path does not have to call
    back into Python during the inner loop.
    """
    if _HAS_NATIVE and _ext is not None:
        views = []
        for e in entries:
            reuse = reuse_lookup(e.session_id) if reuse_lookup is not None else 0.0
            v = _ext.CacheEntryView()
            v.session_id = e.session_id
            v.n_tokens = int(e.n_tokens)
            v.last_access_time = float(e.last_access_time)
            v.predicted_reuse = float(reuse)
            v.pinned = bool(getattr(e, "pinned", False))
            v.ttl_deadline = float(getattr(e, "ttl_deadline", float("inf")))
            views.append(v)
        return int(
            _ext.walru_select_victim(
                views,
                float(now),
                float(max(1.0, tau_max)),
                int(max(1, size_max)),
                float(alpha),
                float(beta),
                float(gamma),
            )
        )

    best_idx = -1
    best_score = float("inf")
    tau = max(1.0, tau_max)
    sz = max(1, size_max)
    for i, e in enumerate(entries):
        if getattr(e, "pinned", False):
            continue
        reuse = reuse_lookup(e.session_id) if reuse_lookup is not None else 0.0
        recency = min(1.0, (now - e.last_access_time) / tau)
        size_norm = min(1.0, e.n_tokens / sz)
        p_evict = alpha * recency + beta * (1.0 - reuse) + gamma * size_norm
        score = -p_evict
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


# --------------------------------------------------------------- Belady


def belady_select_victim(
    entries: Sequence[CacheEntryProtocol],
    now: float,
    future_accesses: Iterable[Sequence[float]],
) -> int:
    """Return the index of the entry whose next access is farthest in the future.

    ``future_accesses[i]`` must be a sorted list of access times for
    ``entries[i]``. An empty list means the entry is never re-accessed
    (best candidate to evict).
    """
    fa_list = [list(x) for x in future_accesses]

    if _HAS_NATIVE and _ext is not None:
        views = []
        for e in entries:
            v = _ext.CacheEntryView()
            v.session_id = e.session_id
            v.n_tokens = int(e.n_tokens)
            v.last_access_time = float(e.last_access_time)
            v.predicted_reuse = 0.0
            v.pinned = bool(getattr(e, "pinned", False))
            v.ttl_deadline = float(getattr(e, "ttl_deadline", float("inf")))
            views.append(v)
        return int(_ext.belady_select_victim(views, float(now), fa_list))

    best_idx = -1
    best_next = -float("inf")
    found_inf = False
    for i, e in enumerate(entries):
        if getattr(e, "pinned", False):
            continue
        times = fa_list[i] if i < len(fa_list) else []
        nxt: float | None = None
        for t in times:
            if t > now:
                nxt = t
                break
        if nxt is None:
            if not found_inf:
                found_inf = True
                best_idx = i
            continue
        if found_inf:
            continue
        if nxt > best_next:
            best_next = nxt
            best_idx = i
    return best_idx


# -------------------------------------------------- predict_reuse_batch


def predict_reuse_batch(
    cached_tokens: Sequence[int],
    succ_probs: Sequence[float],
    succ_obs_tokens: Sequence[int],
    succ_offsets: Sequence[int],
) -> list[float]:
    """Batched ``P_reuse`` over a CSR-encoded successor list.

    For entry *i*, the successor edges are ``succ_probs[lo:hi]`` and
    ``succ_obs_tokens[lo:hi]`` where ``lo = succ_offsets[i]`` and
    ``hi = succ_offsets[i + 1]`` (last entry implicit at end of arrays).
    """
    if _HAS_NATIVE and _ext is not None:
        return list(
            _ext.predict_reuse_batch(
                list(cached_tokens),
                list(succ_probs),
                list(succ_obs_tokens),
                list(succ_offsets),
            )
        )

    n = len(cached_tokens)
    out: list[float] = [0.0] * n
    total = len(succ_probs)
    for i in range(n):
        c = cached_tokens[i]
        if c <= 0:
            continue
        lo = succ_offsets[i] if i < len(succ_offsets) else 0
        hi = succ_offsets[i + 1] if i + 1 < len(succ_offsets) else total
        s = 0.0
        for j in range(lo, hi):
            p = succ_probs[j]
            obs = max(1, succ_obs_tokens[j])
            overlap = c / (c + obs)
            s += p * overlap
        out[i] = max(0.0, min(1.0, s))
    return out


# -------------------------------------------- zero-copy NumPy kernels


def walru_select_victim_flat(
    n_tokens,
    last_access,
    reuse,
    pinned,
    now: float,
    tau_max: float,
    size_max: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> int:
    """Zero-copy WA-LRU victim selection over flat NumPy arrays.

    All four input arrays must have the same length N. ``pinned`` is a
    ``uint8`` array (0/1). The native path takes a direct buffer view; the
    Python fallback uses NumPy's vectorized ops.
    """
    import numpy as np

    if _HAS_NATIVE and _ext is not None:
        return int(
            _ext.walru_select_victim_flat(
                np.ascontiguousarray(n_tokens, dtype=np.int64),
                np.ascontiguousarray(last_access, dtype=np.float64),
                np.ascontiguousarray(reuse, dtype=np.float64),
                np.ascontiguousarray(pinned, dtype=np.uint8),
                float(now),
                float(max(1.0, tau_max)),
                float(max(1.0, size_max)),
                float(alpha),
                float(beta),
                float(gamma),
            )
        )

    n_tokens = np.asarray(n_tokens, dtype=np.float64)
    last_access = np.asarray(last_access, dtype=np.float64)
    reuse = np.asarray(reuse, dtype=np.float64)
    pinned = np.asarray(pinned, dtype=bool)
    tau = max(1.0, tau_max)
    sz = max(1.0, size_max)
    recency = np.minimum(1.0, (now - last_access) / tau)
    size_norm = np.minimum(1.0, n_tokens / sz)
    p_evict = alpha * recency + beta * (1.0 - reuse) + gamma * size_norm
    score = -p_evict
    score = np.where(pinned, np.inf, score)
    if not np.isfinite(score).any():
        return -1
    return int(np.argmin(score))


def belady_select_victim_flat(
    pinned,
    future_times,
    future_offsets,
    now: float,
) -> int:
    """Zero-copy Belady oracle over a CSR-encoded future-access list.

    ``future_times`` is a flat ``float64`` array; entry *i*'s future
    accesses are ``future_times[future_offsets[i]:future_offsets[i+1]]``.
    """
    import numpy as np

    if _HAS_NATIVE and _ext is not None:
        return int(
            _ext.belady_select_victim_flat(
                np.ascontiguousarray(pinned, dtype=np.uint8),
                np.ascontiguousarray(future_times, dtype=np.float64),
                np.ascontiguousarray(future_offsets, dtype=np.int64),
                float(now),
            )
        )

    pi = np.asarray(pinned, dtype=bool)
    ft = np.asarray(future_times, dtype=np.float64)
    fo = np.asarray(future_offsets, dtype=np.int64)
    n = pi.shape[0]
    best_idx = -1
    best_next = -np.inf
    found_inf = False
    for i in range(n):
        if pi[i]:
            continue
        lo = int(fo[i])
        hi = int(fo[i + 1]) if i + 1 < fo.shape[0] else int(ft.shape[0])
        idx = lo + int(np.searchsorted(ft[lo:hi], now, side="right"))
        if idx >= hi:
            if not found_inf:
                found_inf = True
                best_idx = i
            continue
        if found_inf:
            continue
        t_next = float(ft[idx])
        if t_next > best_next:
            best_next = t_next
            best_idx = i
    return best_idx


# --------------------------------------------------- session table


class NativeSessionTable:
    """Sharded session-to-worker map.

    Backed by the C++ ``SessionTable`` when available, by a plain Python
    dict otherwise. The native path uses 64 fine-grained mutex shards.
    """

    def __init__(self, n_shards: int = 64) -> None:
        if _HAS_NATIVE and _ext is not None:
            self._impl = _ext.SessionTable(n_shards)
            self._native = True
        else:
            self._impl = {}
            self._native = False

    def set(self, session_id: str, worker_id: int) -> int:
        if self._native:
            return int(self._impl.set(session_id, worker_id))
        self._impl[session_id] = worker_id
        return worker_id

    def get(self, session_id: str) -> int | None:
        if self._native:
            v = self._impl.get(session_id)
            return None if v is None else int(v)
        return self._impl.get(session_id)

    def erase(self, session_id: str) -> None:
        if self._native:
            self._impl.erase(session_id)
        else:
            self._impl.pop(session_id, None)

    def __len__(self) -> int:
        if self._native:
            return int(self._impl.size())
        return len(self._impl)


# --------------------------------------------------- type hint


class CacheEntryProtocol:
    """Structural type the kernels expect."""

    session_id: str
    n_tokens: int
    last_access_time: float
    pinned: bool


__all__ = [
    "NativeSessionTable",
    "belady_select_victim",
    "belady_select_victim_flat",
    "build_info",
    "is_native_available",
    "predict_reuse_batch",
    "walru_select_victim",
    "walru_select_victim_flat",
]
