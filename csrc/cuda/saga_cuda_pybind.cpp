// pybind11 wrapper for the SAGA CUDA kernels.
//
// Exposes the host-side launchers as Python-callable functions taking
// torch::Tensor inputs. The Python side imports this module as
// ``saga._cuda``; the wrapper in saga.serving.cuda picks it up
// transparently and falls back to a no-op when the build is skipped.

#include "saga_cuda.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <stdexcept>

namespace py = pybind11;

namespace {

void check_cuda_(const torch::Tensor& t, const char* name) {
    if (!t.is_cuda()) {
        throw std::runtime_error(
            std::string(name) + " must be a CUDA tensor");
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(
            std::string(name) + " must be contiguous");
    }
}

saga::cuda::PagedBlockShape unpack_shape_(py::object shape) {
    saga::cuda::PagedBlockShape s;
    s.block_size  = shape.attr("block_size").cast<std::int32_t>();
    s.n_kv_heads  = shape.attr("n_kv_heads").cast<std::int32_t>();
    s.head_dim    = shape.attr("head_dim").cast<std::int32_t>();
    s.dtype_bytes = shape.attr("dtype_bytes").cast<std::int32_t>();
    return s;
}

std::int64_t prefetch_blocks_py(
    torch::Tensor src_k,
    torch::Tensor src_v,
    torch::Tensor dst_k,
    torch::Tensor dst_v,
    torch::Tensor src_block_ids,
    torch::Tensor dst_block_ids,
    py::object shape,
    std::int64_t stream_ptr)
{
    check_cuda_(src_k, "src_k");
    check_cuda_(src_v, "src_v");
    check_cuda_(dst_k, "dst_k");
    check_cuda_(dst_v, "dst_v");

    const auto s = unpack_shape_(shape);
    const std::int32_t n_blocks =
        static_cast<std::int32_t>(src_block_ids.size(0));
    return saga::cuda::prefetch_blocks_launch(
        static_cast<const std::uint16_t*>(src_k.data_ptr()),
        static_cast<const std::uint16_t*>(src_v.data_ptr()),
        static_cast<std::uint16_t*>(dst_k.data_ptr()),
        static_cast<std::uint16_t*>(dst_v.data_ptr()),
        src_block_ids.data_ptr<std::int32_t>(),
        dst_block_ids.data_ptr<std::int32_t>(),
        n_blocks, s,
        reinterpret_cast<void*>(stream_ptr));
}

std::int64_t migration_send_py(
    torch::Tensor k_blocks,
    torch::Tensor v_blocks,
    torch::Tensor src_block_ids,
    std::int32_t peer_rank,
    py::object shape,
    std::int64_t stream_ptr)
{
    check_cuda_(k_blocks, "k_blocks");
    check_cuda_(v_blocks, "v_blocks");
    const auto s = unpack_shape_(shape);
    const std::int32_t n_blocks =
        static_cast<std::int32_t>(src_block_ids.size(0));
    return saga::cuda::migration_send_launch(
        static_cast<const std::uint16_t*>(k_blocks.data_ptr()),
        static_cast<const std::uint16_t*>(v_blocks.data_ptr()),
        src_block_ids.data_ptr<std::int32_t>(),
        n_blocks, peer_rank, s,
        reinterpret_cast<void*>(stream_ptr));
}

std::int64_t migration_recv_py(
    torch::Tensor k_blocks,
    torch::Tensor v_blocks,
    torch::Tensor dst_block_ids,
    std::int32_t peer_rank,
    py::object shape,
    std::int64_t stream_ptr)
{
    check_cuda_(k_blocks, "k_blocks");
    check_cuda_(v_blocks, "v_blocks");
    const auto s = unpack_shape_(shape);
    const std::int32_t n_blocks =
        static_cast<std::int32_t>(dst_block_ids.size(0));
    return saga::cuda::migration_recv_launch(
        static_cast<std::uint16_t*>(k_blocks.data_ptr()),
        static_cast<std::uint16_t*>(v_blocks.data_ptr()),
        dst_block_ids.data_ptr<std::int32_t>(),
        n_blocks, peer_rank, s,
        reinterpret_cast<void*>(stream_ptr));
}

torch::Tensor prefix_overlap_batch_py(
    torch::Tensor tokens_cached,
    torch::Tensor tokens_succ_flat,
    torch::Tensor succ_offsets,
    std::int64_t stream_ptr)
{
    check_cuda_(tokens_cached, "tokens_cached");
    check_cuda_(tokens_succ_flat, "tokens_succ_flat");
    check_cuda_(succ_offsets, "succ_offsets");

    const std::int32_t n_succ =
        static_cast<std::int32_t>(succ_offsets.size(0)) - 1;
    auto out = torch::empty({n_succ}, tokens_cached.options());

    saga::cuda::prefix_overlap_batch_launch(
        tokens_cached.data_ptr<std::int32_t>(),
        static_cast<std::int32_t>(tokens_cached.size(0)),
        tokens_succ_flat.data_ptr<std::int32_t>(),
        succ_offsets.data_ptr<std::int32_t>(),
        n_succ,
        out.data_ptr<std::int32_t>(),
        reinterpret_cast<void*>(stream_ptr));
    return out;
}

py::tuple walru_score_py(
    torch::Tensor recency,
    torch::Tensor preuse,
    torch::Tensor size_norm,
    torch::Tensor pinned,
    float alpha, float beta, float gamma,
    std::int64_t stream_ptr)
{
    check_cuda_(recency, "recency");
    check_cuda_(preuse, "preuse");
    check_cuda_(size_norm, "size_norm");
    check_cuda_(pinned, "pinned");

    const std::int32_t n =
        static_cast<std::int32_t>(recency.size(0));
    auto out_argmin    = torch::empty({1}, recency.options().dtype(torch::kInt32));
    auto out_min_score = torch::empty({1}, recency.options());

    saga::cuda::walru_score_launch(
        recency.data_ptr<float>(),
        preuse.data_ptr<float>(),
        size_norm.data_ptr<float>(),
        static_cast<const std::uint8_t*>(pinned.data_ptr()),
        n, alpha, beta, gamma,
        out_argmin.data_ptr<std::int32_t>(),
        out_min_score.data_ptr<float>(),
        reinterpret_cast<void*>(stream_ptr));

    return py::make_tuple(out_argmin, out_min_score);
}

std::int32_t compact_pool_py(
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor block_table,
    torch::Tensor alive_mask,
    py::object shape,
    std::int64_t stream_ptr)
{
    check_cuda_(k_pool, "k_pool");
    check_cuda_(v_pool, "v_pool");
    check_cuda_(block_table, "block_table");
    check_cuda_(alive_mask, "alive_mask");

    const auto s = unpack_shape_(shape);
    const std::int32_t n_physical =
        static_cast<std::int32_t>(alive_mask.size(0));
    return saga::cuda::compact_pool_launch(
        static_cast<std::uint16_t*>(k_pool.data_ptr()),
        static_cast<std::uint16_t*>(v_pool.data_ptr()),
        block_table.data_ptr<std::int32_t>(),
        static_cast<const std::uint8_t*>(alive_mask.data_ptr()),
        n_physical, s,
        reinterpret_cast<void*>(stream_ptr));
}

}  // anonymous namespace

PYBIND11_MODULE(_cuda, m) {
    m.doc() = "SAGA CUDA kernels (prefetch, migration, overlap, WA-LRU "
              "scoring, pool compaction). Loaded as saga._cuda; the wrapper "
              "in saga.serving.cuda falls back to the CPU paths when this "
              "module is not built.";

    m.def("prefetch_blocks", &prefetch_blocks_py,
          py::arg("src_k"), py::arg("src_v"),
          py::arg("dst_k"), py::arg("dst_v"),
          py::arg("src_block_ids"), py::arg("dst_block_ids"),
          py::arg("shape"), py::arg("stream_ptr") = 0);

    m.def("migration_send", &migration_send_py,
          py::arg("k_blocks"), py::arg("v_blocks"),
          py::arg("src_block_ids"), py::arg("peer_rank"),
          py::arg("shape"), py::arg("stream_ptr") = 0);

    m.def("migration_recv", &migration_recv_py,
          py::arg("k_blocks"), py::arg("v_blocks"),
          py::arg("dst_block_ids"), py::arg("peer_rank"),
          py::arg("shape"), py::arg("stream_ptr") = 0);

    m.def("prefix_overlap_batch", &prefix_overlap_batch_py,
          py::arg("tokens_cached"), py::arg("tokens_succ_flat"),
          py::arg("succ_offsets"), py::arg("stream_ptr") = 0);

    m.def("walru_score", &walru_score_py,
          py::arg("recency"), py::arg("preuse"),
          py::arg("size_norm"), py::arg("pinned"),
          py::arg("alpha"), py::arg("beta"), py::arg("gamma"),
          py::arg("stream_ptr") = 0);

    m.def("compact_pool", &compact_pool_py,
          py::arg("k_pool"), py::arg("v_pool"),
          py::arg("block_table"), py::arg("alive_mask"),
          py::arg("shape"), py::arg("stream_ptr") = 0);
}
