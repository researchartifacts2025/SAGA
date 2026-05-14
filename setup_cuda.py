"""Builder for the optional SAGA CUDA extension.

Run after ``setup_native.py`` if you have an NVIDIA GPU with CUDA 12.1+
installed and want the GPU paths active:

    pip install "pybind11>=2.11" "torch>=2.1.2"
    python setup_cuda.py build_ext --inplace

The resulting shared object lands at ``src/saga/_cuda.<platform>.{so,pyd}``;
:mod:`saga.serving.cuda` discovers it automatically. If the build is skipped
or fails (no CUDA toolchain), the Python-level scheduler still works -- the
CUDA kernels are accelerators, not requirements.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    print(
        "ERROR: torch is required to build the SAGA CUDA extension.\n"
        "Install it with: pip install 'torch>=2.1.2'",
        file=sys.stderr,
    )
    raise SystemExit(1) from None


if not torch.cuda.is_available() and "FORCE_CUDA" not in os.environ:
    print(
        "[saga-cuda] CUDA is not available at build-host runtime; building\n"
        "  with the toolchain anyway (set FORCE_CUDA=0 to abort instead).",
        file=sys.stderr,
    )


HERE = Path(__file__).parent
CUDA_SRC = HERE / "csrc" / "cuda"

# A100 (sm_80) is the primary target per the paper; we also emit sm_70
# (V100) and sm_90 (H100) so binary wheels work across the common HPC
# fleet.
ARCH_FLAGS = [
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_90,code=sm_90",
]


from setuptools import setup  # noqa: E402  (after the torch import guard)


ext = CUDAExtension(
    name="saga._cuda",
    sources=[
        str(CUDA_SRC / "prefetch_stream.cu"),
        str(CUDA_SRC / "migration.cu"),
        str(CUDA_SRC / "prefix_overlap.cu"),
        str(CUDA_SRC / "walru_score_cuda.cu"),
        str(CUDA_SRC / "compact_pool.cu"),
        str(CUDA_SRC / "saga_cuda_pybind.cpp"),
    ],
    include_dirs=[str(CUDA_SRC)],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--extended-lambda",
            "-std=c++17",
        ]
        + ARCH_FLAGS,
    },
)


if __name__ == "__main__":
    setup(
        name="saga-cuda-shim",
        version="0.0.0",
        description="Build shim for the optional SAGA CUDA extension.",
        ext_modules=[ext],
        packages=["saga"],
        package_dir={"saga": "src/saga"},
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
