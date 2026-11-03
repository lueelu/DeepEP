"""Upstream-named interfaces with compatibility aliases over the direct MTE/UDMA implementation."""

from .buffer import Buffer as SourceBuffer
from .elastic import ElasticBuffer as SourceElasticBuffer
from ..._compat import keyword_aliases


class Buffer(SourceBuffer):
    __init__ = keyword_aliases(
        SourceBuffer.__init__,
        {
            "num_scaleup_bytes": "num_nvl_bytes",
            "num_scaleout_bytes": "num_rdma_bytes",
            "allow_memory_transport_for_low_latency_mode": "allow_nvlink_for_low_latency_mode",
            "allow_multinode_memory_transport": "allow_mnnvl",
        },
    )
    dispatch = keyword_aliases(SourceBuffer.dispatch, {"num_tokens_per_scaleout_rank": "num_tokens_per_rdma_rank"})
    set_num_sms = staticmethod(keyword_aliases(SourceBuffer.set_num_sms, {"new_num_compute_units": "new_num_sms"}))
    set_num_compute_units = set_num_sms


class ElasticBuffer(SourceElasticBuffer):
    __init__ = keyword_aliases(SourceElasticBuffer.__init__, {"num_device_timeout_secs": "num_gpu_timeout_secs"})
    dispatch = keyword_aliases(
        SourceElasticBuffer.dispatch,
        {"num_compute_units": "num_sms", "use_aligned_col_major_sf": "use_tma_aligned_col_major_sf"},
    )
    combine = keyword_aliases(SourceElasticBuffer.combine, {"num_compute_units": "num_sms"})
    get_theoretical_num_sms = keyword_aliases(
        SourceElasticBuffer.get_theoretical_num_sms,
        {
            "scaleup_gbs": "nvlink_gbs",
            "scaleout_gbs": "rdma_gbs",
            "compute_unit_read_gbs": "sm_read_gbs",
            "compute_unit_write_gbs": "sm_write_gbs",
            "compute_read_gbs": "sm_read_gbs",
            "compute_write_gbs": "sm_write_gbs",
        },
    )
    get_theoretical_num_compute_units = get_theoretical_num_sms
    get_theoretical_num_qps = keyword_aliases(
        SourceElasticBuffer.get_theoretical_num_qps, {"num_compute_units": "num_sms"}
    )


def buffer_class(name):
    return {"Buffer": Buffer, "ElasticBuffer": ElasticBuffer}[name]
