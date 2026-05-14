"""Standalone builder for the optional ``saga._native`` extension.

Use this when you do not want CMake. Run

    pip install pybind11>=2.11
    python setup_native.py build_ext --inplace

The resulting shared object (``_native.<platform>.so`` or ``.pyd``) lands in
``src/saga/``, where :mod:`saga.native` will discover it automatically.

The main ``pip install -e .`` flow does NOT require this. SAGA is a pure
Python package by default; the native module is a transparent accelerator
for the hot WA-LRU and Belady paths.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from setuptools import setup, Extension

try:
    import pybind11  # type: ignore[import-not-found]
except ImportError as exc:
    print(
        "ERROR: pybind11 is required to build the native extension.\n"
        "Install it with: pip install 'pybind11>=2.11'",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


HERE = Path(__file__).parent
CSRC = HERE / "csrc"


def _extra_compile_args() -> list[str]:
    if platform.system() == "Windows":
        return ["/O2", "/std:c++17", "/openmp"]
    args = ["-O3", "-std=c++17", "-fvisibility=hidden"]
    if platform.machine() in ("x86_64", "AMD64"):
        args.append("-march=native")
    return args


def _extra_link_args() -> list[str]:
    if platform.system() == "Windows":
        return []
    return ["-fopenmp"] if _has_openmp() else []


def _has_openmp() -> bool:
    if platform.system() == "Darwin":
        # libomp must be installed; conservatively disable to avoid noise.
        return False
    return True


def _compile_with_openmp_flag() -> list[str]:
    if platform.system() == "Windows":
        return []
    return ["-fopenmp"] if _has_openmp() else []


ext = Extension(
    name="saga._native",
    sources=[str(CSRC / "saga_native.cpp")],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=_extra_compile_args() + _compile_with_openmp_flag(),
    extra_link_args=_extra_link_args(),
)


if __name__ == "__main__":
    # We pass ``packages=["saga"]`` and ``package_dir={"saga": "src/saga"}`` so
    # setuptools' ``build_ext --inplace`` places the compiled module at
    # ``src/saga/_native.<platform>.{so,pyd,dylib}``. Earlier versions used the
    # shorthand ``package_dir={"": "src"}`` which combined with empty-package
    # auto-discovery could flatten the ``src/saga/`` tree on some setuptools
    # versions; the explicit form below avoids that.
    setup(
        name="saga-native-shim",
        version="0.0.0",
        description="Build shim for the optional SAGA native extension.",
        ext_modules=[ext],
        packages=["saga"],
        package_dir={"saga": "src/saga"},
        zip_safe=False,
    )
