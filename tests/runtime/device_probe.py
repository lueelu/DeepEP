# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Explicit torchrun qualification of resource lifecycle and real event ordering.

This is not an EP communication or performance test. Run only on reserved NPUs.
"""

from datetime import timedelta
import argparse
import json
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from deep_ep import Buffer, ElasticBuffer, EventHandle, EventOverlap
from deep_ep._native import require_native


def expect(error, action):
    try:
        action()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hccl", "gloo"], default="hccl")
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(args.backend, timeout=timedelta(seconds=180))
    group = dist.group.WORLD
    try:
        # Actual stream dependency, not just a successful event method call.
        producer = torch.npu.Stream()
        with torch.npu.stream(producer):
            value = torch.ones(65536, device="npu")
            value.mul_(7)
            event = EventHandle()
        overlap = EventOverlap(event, (value,))
        calls = []
        overlap.register_hook_after_wait(lambda: calls.append("waited"))
        overlap.current_stream_wait(release_handle=True)
        assert torch.equal(value.cpu(), torch.full((65536,), 7.0))
        assert calls == ["waited"] and overlap.extra_tensors is None
        expect(RuntimeError, overlap.current_stream_wait)
        expect(ValueError, lambda: EventOverlap(None))
        native = require_native("device probe")
        # Both generations, repeated lifetimes, rejected concurrent ownership.
        for generation in (Buffer, ElasticBuffer, Buffer, ElasticBuffer):
            kwargs = {
                "num_scaleup_bytes" if generation is Buffer else "num_bytes": 4 << 20,
                "explicitly_destroy": True,
            }
            buf = generation(group, **kwargs)
            try:
                assert native.runtime_busy()
                assert buf.get_comm_stream() is not None
                with torch.npu.stream(buf.get_comm_stream()):
                    surviving_tensor = torch.ones(65536, device="npu")
                    surviving_tensor.mul_(3)
                    with EventOverlap(EventHandle(), (surviving_tensor,))(True):
                        pass
                expect(RuntimeError, lambda: generation(group, **kwargs))
                expect(ValueError if generation is ElasticBuffer else NotImplementedError, lambda: buf.dispatch(None))
                if generation is ElasticBuffer:
                    expect(NotImplementedError, buf.barrier)
                    buf.barrier(with_cpu_sync=True)
                captured = buf.capture()
                captured.current_stream_wait()
            finally:
                buf.destroy()
            buf.destroy()
            assert torch.equal(surviving_tensor.cpu(), torch.full((65536,), 3.0))
            assert not native.runtime_busy()
            expect(RuntimeError, buf.get_comm_stream)
            assert dist.is_initialized()  # Library never destroys caller's PG.
        expect(
            ValueError,
            lambda: ElasticBuffer(
                group,
                num_bytes=(2 + dist.get_rank()) * (2 << 20),
                explicitly_destroy=True,
            ),
        )
        assert not native.runtime_busy()
        torch.npu.synchronize()
        print(
            json.dumps(
                {
                    "rank": dist.get_rank(),
                    "world_size": dist.get_world_size(),
                    "bootstrap_backend": args.backend,
                    "runtime_lifetimes": 4,
                    "event_ordering": "passed",
                    "ep_operations": "not implemented",
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
