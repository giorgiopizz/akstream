"""the C++ extension akstream._hep (the rest of the package is pure Python, configured in pyproject.toml).

Compiled without fused multiply-add and without the sin + cos -> sincos fusion, so that the kernels round as the numpy
formulas they replace. AKSTREAM_STATIC_LIBSTDCXX=1 links libstdc++ statically (for a build used where the compiler
runtime is absent).
"""
from pybind11.setup_helpers import Pybind11Extension
import os

from setuptools import setup

# AKSTREAM_STATIC_LIBSTDCXX=1: link libstdc++ statically (a build that loads where the compiler's runtime is absent)
EXTRA_LINK = ["-static-libstdc++", "-static-libgcc"] if os.environ.get("AKSTREAM_STATIC_LIBSTDCXX") == "1" else []

setup(ext_modules=[Pybind11Extension(
    "akstream._hep", ["akstream/_hep.cpp"], cxx_std=17,
    # no fused multiply-add, and no sin + cos -> sincos fusion (glibc's sincos rounds differently from sin and cos):
    # the kernels must round exactly as the numpy formulas they replace
    extra_compile_args=["-O2", "-ffp-contract=off", "-fno-fast-math", "-fno-builtin-sin", "-fno-builtin-cos",
                        "-fno-builtin-sincos"],
    extra_link_args=EXTRA_LINK,
)])
