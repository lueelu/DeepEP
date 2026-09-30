# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by zhu-mingzhe71 2026
"""Ascend950 fused communication/GMM operators; optional native wheel component."""

from importlib import import_module
from ._build_versions import (
    CATCCOS_COMMIT as CATCCOS_COMMIT,
    SHMEM_COMMIT as SHMEM_COMMIT,
    SHMEM_REF as SHMEM_REF,
    SHMEM_VERSION as SHMEM_VERSION,
)
from ._version import __version__ as __version__

_EXPORTS = {
    "ops": [
        "alltoallv_gmm",
        "gmm_alltoallv",
        "all_to_allv_grouped_matmul",
        "grouped_matmul_all_to_allv",
    ],
    "runtime": [
        "UdmaConfig",
        "destroy_udma_group",
        "get_udma_config",
        "init_udma_group",
        "udma_group",
    ],
    "routing": ["build_global_route_matrix", "build_rank_route", "build_rank_route_tables", "build_rank_route_tensors"],
    "tiling": ["load_tiling_config"],
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f"{__name__}.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(name)
