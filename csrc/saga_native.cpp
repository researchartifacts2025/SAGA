// SAGA native acceleration kernels.
//
// The pure-Python policy implementations are correct but bottleneck on large
// caches: WA-LRU has to score every resident entry, and Belady has to scan
// each entry's future-access list. These kernels move those hot loops into
// C++ with optional OpenMP parallelism.
//
// Loaded by saga.native as the optional `_native` module. If the build is
// skipped, saga.native falls back to the pure-Python paths transparently.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

struct CacheEntryView {
    std::string session_id;
    std::int64_t n_tokens;
    double last_access_time;
    double predicted_reuse;
    bool pinned;
    double ttl_deadline;
};

inline double walru_score(
    const CacheEntryView& e,
    double now,
    double tau_max,
    double size_max,
    double alpha, double beta, double gamma) noexcept
{
    const double tau = std::max(1.0, tau_max);
    const double sz  = std::max(1.0, size_max);
    const double recency = std::min(1.0, (now - e.last_access_time) / tau);
    const double size_norm = std::min(1.0, double(e.n_tokens) / sz);
    const double p_evict = alpha * recency
                         + beta  * (1.0 - e.predicted_reuse)
                         + gamma * size_norm;
    return -p_evict;
}

// ---------------------------------------------------------------- WA-LRU
//
// Pick the entry with the smallest score (largest p_evict). Pinned entries
// are skipped. Parallel reduction via OpenMP when available; the cost of
// thread creation only pays off above a few hundred entries.

int walru_select_victim(
    const std::vector<CacheEntryView>& entries,
    double now,
    double tau_max,
    double size_max,
    double alpha, double beta, double gamma)
{
    const int n = static_cast<int>(entries.size());
    if (n == 0) return -1;

    int best_idx = -1;
    double best_score = std::numeric_limits<double>::infinity();

#if defined(_OPENMP)
    if (n >= 256) {
        int  local_idx   = -1;
        double local_min = std::numeric_limits<double>::infinity();
        #pragma omp parallel
        {
            int  t_idx = -1;
            double t_min = std::numeric_limits<double>::infinity();
            #pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                if (entries[i].pinned) continue;
                const double s = walru_score(entries[i], now, tau_max, size_max,
                                             alpha, beta, gamma);
                if (s < t_min) { t_min = s; t_idx = i; }
            }
            #pragma omp critical
            {
                if (t_min < local_min) { local_min = t_min; local_idx = t_idx; }
            }
        }
        best_idx = local_idx;
        return best_idx;
    }
#endif

    for (int i = 0; i < n; ++i) {
        if (entries[i].pinned) continue;
        const double s = walru_score(entries[i], now, tau_max, size_max,
                                     alpha, beta, gamma);
        if (s < best_score) { best_score = s; best_idx = i; }
    }
    return best_idx;
}

// ---------------- WA-LRU on flat NumPy arrays ----------------
//
// The CacheEntryView path is convenient but pays an O(N) per-call
// marshalling cost. The hot path takes flat NumPy arrays (one per field),
// which pybind11 hands us as a zero-copy buffer; no per-entry allocation.

int walru_select_victim_flat(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> n_tokens,
    py::array_t<double, py::array::c_style | py::array::forcecast>       last_access,
    py::array_t<double, py::array::c_style | py::array::forcecast>       reuse,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> pinned,
    double now,
    double tau_max,
    double size_max,
    double alpha, double beta, double gamma)
{
    auto t  = n_tokens.unchecked<1>();
    auto la = last_access.unchecked<1>();
    auto re = reuse.unchecked<1>();
    auto pi = pinned.unchecked<1>();

    const int n = static_cast<int>(t.shape(0));
    if (n == 0) return -1;

    const double tau = std::max(1.0, tau_max);
    const double sz  = std::max(1.0, size_max);

#if defined(_OPENMP)
    if (n >= 256) {
        int  best_idx = -1;
        double best_score = std::numeric_limits<double>::infinity();
        #pragma omp parallel
        {
            int    local_idx = -1;
            double local_min = std::numeric_limits<double>::infinity();
            #pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                if (pi(i)) continue;
                const double recency = std::min(1.0, (now - la(i)) / tau);
                const double size_n  = std::min(1.0, double(t(i)) / sz);
                const double p_evict = alpha * recency
                                     + beta  * (1.0 - re(i))
                                     + gamma * size_n;
                const double score = -p_evict;
                if (score < local_min) { local_min = score; local_idx = i; }
            }
            #pragma omp critical
            {
                if (local_min < best_score) {
                    best_score = local_min;
                    best_idx = local_idx;
                }
            }
        }
        return best_idx;
    }
#endif

    int best_idx = -1;
    double best_score = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        if (pi(i)) continue;
        const double recency = std::min(1.0, (now - la(i)) / tau);
        const double size_n  = std::min(1.0, double(t(i)) / sz);
        const double p_evict = alpha * recency
                             + beta  * (1.0 - re(i))
                             + gamma * size_n;
        const double score = -p_evict;
        if (score < best_score) { best_score = score; best_idx = i; }
    }
    return best_idx;
}

// Belady oracle with CSR-encoded future-access list.
//
// future_times[future_offsets[i] : future_offsets[i+1]] are sorted, future
// access times for entry i. This avoids the per-entry std::vector copy of
// the CacheEntryView path.

int belady_select_victim_flat(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> pinned,
    py::array_t<double, py::array::c_style | py::array::forcecast>       future_times,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> future_offsets,
    double now)
{
    auto pi = pinned.unchecked<1>();
    auto ft = future_times.unchecked<1>();
    auto fo = future_offsets.unchecked<1>();

    const int n = static_cast<int>(pi.shape(0));
    if (n == 0) return -1;

    int best_idx = -1;
    double best_next = -std::numeric_limits<double>::infinity();
    bool best_has_inf = false;

    for (int i = 0; i < n; ++i) {
        if (pi(i)) continue;
        const std::int64_t lo = fo(i);
        const std::int64_t hi = (i + 1 < fo.shape(0))
                              ? fo(i + 1)
                              : static_cast<std::int64_t>(ft.shape(0));
        // Binary search for the first timestamp strictly greater than `now`.
        std::int64_t l = lo, r = hi;
        while (l < r) {
            const std::int64_t m = (l + r) / 2;
            if (ft(m) > now) r = m; else l = m + 1;
        }
        if (l >= hi) {
            if (!best_has_inf) { best_has_inf = true; best_idx = i; }
            continue;
        }
        if (best_has_inf) continue;
        const double t_next = ft(l);
        if (t_next > best_next) { best_next = t_next; best_idx = i; }
    }
    return best_idx;
}

// Score *every* entry; useful when a benchmark needs a full priority order.

std::vector<double> walru_score_batch(
    const std::vector<CacheEntryView>& entries,
    double now,
    double tau_max,
    double size_max,
    double alpha, double beta, double gamma)
{
    const int n = static_cast<int>(entries.size());
    std::vector<double> out(n);
#if defined(_OPENMP)
    #pragma omp parallel for if(n >= 256)
#endif
    for (int i = 0; i < n; ++i) {
        out[i] = walru_score(entries[i], now, tau_max, size_max,
                             alpha, beta, gamma);
    }
    return out;
}

// ------------------------------------------------------------- Belady
//
// Each entry has its own sorted future-access list. The oracle evicts the
// entry whose next access is farthest in the future (or never). Binary
// search per entry; O(N log K) instead of Python's O(N * K).

int belady_select_victim(
    const std::vector<CacheEntryView>& entries,
    double now,
    const std::vector<std::vector<double>>& future_accesses)
{
    const int n = static_cast<int>(entries.size());
    if (n == 0) return -1;

    int best_idx = -1;
    double best_next = -std::numeric_limits<double>::infinity();
    bool best_has_inf = false;

    for (int i = 0; i < n; ++i) {
        if (entries[i].pinned) continue;
        const auto& times = (i < static_cast<int>(future_accesses.size()))
                          ? future_accesses[i]
                          : std::vector<double>{};
        auto it = std::upper_bound(times.begin(), times.end(), now);
        if (it == times.end()) {
            if (!best_has_inf) {
                best_has_inf = true;
                best_idx = i;
            }
            continue;
        }
        if (best_has_inf) continue;
        const double t = *it;
        if (t > best_next) { best_next = t; best_idx = i; }
    }
    return best_idx;
}

// ------------------------------------------------- lock-free session table
//
// A small concurrent hash table backing the global coordinator's
// session-to-worker affinity map. Uses fine-grained shard locks rather than
// a single global mutex; the API surface matches the Python dict-style use.
// This is the C++ analogue of paper §3.7's lock-free CAS table.

class SessionTable {
public:
    explicit SessionTable(std::size_t n_shards = 64)
        : shards_(n_shards) {}

    int set(const std::string& sid, int worker_id) {
        auto& s = shard_for(sid);
        std::lock_guard<std::mutex> lk(s.mu);
        s.map[sid] = worker_id;
        return worker_id;
    }

    py::object get(const std::string& sid) {
        auto& s = shard_for(sid);
        std::lock_guard<std::mutex> lk(s.mu);
        auto it = s.map.find(sid);
        if (it == s.map.end()) return py::none();
        return py::int_(it->second);
    }

    void erase(const std::string& sid) {
        auto& s = shard_for(sid);
        std::lock_guard<std::mutex> lk(s.mu);
        s.map.erase(sid);
    }

    std::size_t size() const {
        std::size_t total = 0;
        for (const auto& s : shards_) {
            std::lock_guard<std::mutex> lk(s.mu);
            total += s.map.size();
        }
        return total;
    }

private:
    struct Shard {
        mutable std::mutex mu;
        std::unordered_map<std::string, int> map;
        Shard() = default;
        Shard(const Shard&) = delete;
        Shard& operator=(const Shard&) = delete;
        Shard(Shard&&) noexcept {}
    };

    Shard& shard_for(const std::string& key) {
        const std::size_t h = std::hash<std::string>{}(key);
        return shards_[h % shards_.size()];
    }

    mutable std::vector<Shard> shards_;
};

// --------------------------------------------------------- predict_reuse
//
// AEG-driven reuse prediction (paper §4.1):
//
//   P_reuse(s) = Σ_u P(v_s → u) · overlap(s, u)
//   overlap(s, u) = cached / (cached + Ê[obs_tokens(u)])
//
// This kernel evaluates a *batch* of (cached_tokens, successor_probs,
// successor_obs_tokens) triples in parallel. The Python side flattens the
// successor lists into a single contiguous array plus a CSR-style offsets
// vector; the kernel does the prefix overlap and weighted sum.

std::vector<double> predict_reuse_batch(
    const std::vector<std::int64_t>& cached_tokens,
    const std::vector<double>& succ_probs,
    const std::vector<std::int64_t>& succ_obs_tokens,
    const std::vector<std::int64_t>& succ_offsets)
{
    const int n = static_cast<int>(cached_tokens.size());
    std::vector<double> out(n, 0.0);

#if defined(_OPENMP)
    #pragma omp parallel for if(n >= 128)
#endif
    for (int i = 0; i < n; ++i) {
        const std::int64_t c = cached_tokens[i];
        if (c <= 0) { out[i] = 0.0; continue; }
        const std::int64_t lo = (i < static_cast<int>(succ_offsets.size()))
                              ? succ_offsets[i] : 0;
        const std::int64_t hi = (i + 1 < static_cast<int>(succ_offsets.size()))
                              ? succ_offsets[i + 1]
                              : static_cast<std::int64_t>(succ_probs.size());
        double sum = 0.0;
        for (std::int64_t j = lo; j < hi; ++j) {
            const double p = succ_probs[j];
            const std::int64_t obs = std::max<std::int64_t>(1, succ_obs_tokens[j]);
            const double overlap = static_cast<double>(c) /
                                   static_cast<double>(c + obs);
            sum += p * overlap;
        }
        if (sum < 0.0) sum = 0.0;
        if (sum > 1.0) sum = 1.0;
        out[i] = sum;
    }
    return out;
}

// ------------------------------------------------------------------- info

std::string build_info() {
#if defined(_OPENMP)
    return std::string("saga_native v1 (OpenMP, threads=")
         + std::to_string(omp_get_max_threads()) + ")";
#else
    return std::string("saga_native v1 (single-threaded)");
#endif
}

} // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "SAGA native acceleration kernels (WA-LRU, Belady, "
              "session table). Optional; the Python fallback is identical "
              "in behavior.";

    py::class_<CacheEntryView>(m, "CacheEntryView")
        .def(py::init<>())
        .def_readwrite("session_id", &CacheEntryView::session_id)
        .def_readwrite("n_tokens", &CacheEntryView::n_tokens)
        .def_readwrite("last_access_time", &CacheEntryView::last_access_time)
        .def_readwrite("predicted_reuse", &CacheEntryView::predicted_reuse)
        .def_readwrite("pinned", &CacheEntryView::pinned)
        .def_readwrite("ttl_deadline", &CacheEntryView::ttl_deadline);

    m.def("walru_select_victim", &walru_select_victim,
          py::arg("entries"), py::arg("now"),
          py::arg("tau_max"), py::arg("size_max"),
          py::arg("alpha"), py::arg("beta"), py::arg("gamma"));

    m.def("walru_score_batch", &walru_score_batch,
          py::arg("entries"), py::arg("now"),
          py::arg("tau_max"), py::arg("size_max"),
          py::arg("alpha"), py::arg("beta"), py::arg("gamma"));

    m.def("belady_select_victim", &belady_select_victim,
          py::arg("entries"), py::arg("now"),
          py::arg("future_accesses"));

    m.def("predict_reuse_batch", &predict_reuse_batch,
          py::arg("cached_tokens"), py::arg("succ_probs"),
          py::arg("succ_obs_tokens"), py::arg("succ_offsets"),
          "Compute P_reuse for a batch of cache entries in parallel.");

    m.def("walru_select_victim_flat", &walru_select_victim_flat,
          py::arg("n_tokens"), py::arg("last_access"),
          py::arg("reuse"), py::arg("pinned"),
          py::arg("now"), py::arg("tau_max"), py::arg("size_max"),
          py::arg("alpha"), py::arg("beta"), py::arg("gamma"),
          "Zero-copy WA-LRU victim selection over flat NumPy arrays.");

    m.def("belady_select_victim_flat", &belady_select_victim_flat,
          py::arg("pinned"), py::arg("future_times"),
          py::arg("future_offsets"), py::arg("now"),
          "Zero-copy Belady oracle over a CSR-encoded future-access list.");

    py::class_<SessionTable>(m, "SessionTable")
        .def(py::init<std::size_t>(), py::arg("n_shards") = 64)
        .def("set", &SessionTable::set)
        .def("get", &SessionTable::get)
        .def("erase", &SessionTable::erase)
        .def("size", &SessionTable::size);

    m.def("build_info", &build_info);
}
