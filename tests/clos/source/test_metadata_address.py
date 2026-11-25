"""Device regression: reserve the old SHMEM Host VA before ElasticBuffer init.

Run as a standalone Linux script. Every invocation uses fresh rank processes.
This checks initialization and a device barrier, not dispatch/combine precision.
"""

import argparse
import ctypes
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import sys


# SHMEM 3d40c07: device end - 1 GiB - 32 MiB metadata/context region.
OLD_META_ADDRESS = 0x180000000000 - (1 << 30) - (32 << 20)
META_BYTES = 32 << 20


class HostReservation:
    def __init__(self):
        if sys.platform != "linux" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("Address injection requires 64-bit Linux")
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.munmap.restype = ctypes.c_int
        # PROT_NONE reserves Host VA without allocating/touching 32 MiB of RAM.
        # NOREPLACE must never overwrite an existing application mapping.
        address = self.libc.mmap(OLD_META_ADDRESS, META_BYTES, 0, 0x02 | 0x20 | 0x100000, -1, 0)
        if address == ctypes.c_void_p(-1).value:
            error = ctypes.get_errno()
            raise OSError(error, f"Host mmap injection failed: {os.strerror(error)}")
        if address != OLD_META_ADDRESS:
            self.libc.munmap(address, META_BYTES)
            raise RuntimeError("Kernel did not honor MAP_FIXED_NOREPLACE; injection is invalid")
        self.address = address

    def close(self):
        if self.address is not None:
            if self.libc.munmap(self.address, META_BYTES):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            self.address = None


def expected_init_error(error, args):
    # Bootstrap, mmap, import failures, crashes and timeouts must not pass the
    # negative case. SHMEM logs still establish the underlying address error.
    return args.expect_init == "failure" and str(error).startswith(
        ("CANN SHMEM call failed: aclshmemx_init_attr(", "CANN SHMEM call failed: aclshmemx_init_attr_with_buffers(")
    )


def save_report(args, report):
    (Path(args.output_dir) / f"rank{report['rank']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_rank(rank, args):
    signal.alarm(args.timeout)
    report = dict(
        rank=rank,
        pid=os.getpid(),
        metadata_mode=args.metadata_mode,
        occupy_old_meta=args.occupy_old_meta,
        expected=args.expect_init,
        old_meta_address=hex(OLD_META_ADDRESS),
        bytes=META_BYTES,
        passed=False,
    )
    reservation = None
    owner = None
    try:
        report["stage"] = "host_mmap"
        if args.occupy_old_meta:
            reservation = HostReservation()
            report["reserved_address"] = hex(reservation.address)
            print(f"HOST_VA_OCCUPIED rank={rank} address={reservation.address:#x} bytes={META_BYTES}", flush=True)

        report["stage"] = "framework_and_process_group"
        save_report(args, report)
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401
        import deep_ep

        device = args.first_device + rank
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(device), WORLD_SIZE=str(args.ep_size))
        torch.npu.set_device(device)
        info = json.loads((Path(deep_ep.__file__).parent / "_build_info.json").read_text())
        report["build_info"] = info
        actual = info.get("shmem", {}).get("metadata_mode", "fixed")
        if actual != args.metadata_mode:
            raise RuntimeError(f"Expected {args.metadata_mode} wheel, loaded {actual}")
        dist.init_process_group(
            "hccl",
            init_method=args.dist_init_method,
            rank=rank,
            world_size=args.ep_size,
            timeout=timedelta(seconds=args.timeout),
        )
        size = deep_ep.ElasticBuffer.get_buffer_size_hint(dist.group.WORLD, 128, 2816, 8, machine_type="global_clos")
        report["stage"] = "buffer_init"
        save_report(args, report)
        try:
            owner = deep_ep.ElasticBuffer(
                dist.group.WORLD,
                size,
                hidden=2816,
                num_topk=8,
                num_max_tokens_per_rank=128,
                machine_type="global_clos",
                explicitly_destroy=True,
            )
        except RuntimeError as error:
            report["init_error"] = str(error)
            if not expected_init_error(error, args):
                raise
            report["result"] = "expected_shmem_init_failure"
        else:
            if args.expect_init != "success":
                raise AssertionError("Fixed-address initialization unexpectedly succeeded; issue was not reproduced")
            report["stage"] = "device_barrier"
            save_report(args, report)
            owner.barrier()
            torch.npu.synchronize()
            owner.destroy()
            owner = None
            report["result"] = "buffer_init_and_barrier_passed"
        # All ranks must reach their expected outcome. A hung peer is a test
        # failure, never a successful reproduction of the address conflict.
        report["stage"] = "process_group_cleanup"
        dist.destroy_process_group()
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if owner is not None:
                owner.destroy()
            if reservation is not None:
                reservation.close()
        except BaseException as error:
            report["passed"] = False
            report["cleanup_error"] = repr(error)
            raise
        finally:
            save_report(args, report)
            print("META_ADDRESS_RESULT " + json.dumps(report), flush=True)
            signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-mode", choices=("fixed", "dynamic"), required=True)
    parser.add_argument("--expect-init", choices=("success", "failure"), required=True)
    parser.add_argument("--occupy-old-meta", action="store_true")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--first-device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dist-init-method", default="tcp://127.0.0.1:29816")
    parser.add_argument("--output-dir", required=True, help="New directory for per-rank results")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This device test requires Linux")
    if args.ep_size < 2 or args.first_device < 0 or args.timeout < 1:
        parser.error("Require EP >= 2, first-device >= 0 and timeout > 0")
    if args.expect_init == "failure" and (args.metadata_mode != "fixed" or not args.occupy_old_meta):
        parser.error("Expected failure is only valid for fixed mode with address injection")
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("SHMEM_LOG_LEVEL", "DEBUG")
    os.environ.setdefault("SHMEM_LOG_TO_STDOUT", "1")
    import torch.multiprocessing as mp

    mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)
    reports = [json.loads((Path(args.output_dir) / f"rank{rank}.json").read_text()) for rank in range(args.ep_size)]
    if not all(report["passed"] for report in reports):
        raise RuntimeError("Not all ranks passed")
    print("META_ADDRESS_TEST_PASSED", flush=True)


if __name__ == "__main__":
    main()
