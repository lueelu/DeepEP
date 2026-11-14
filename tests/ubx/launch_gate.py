# Copyright (c) 2026, Lu Lu
"""Benchmark-only ACL Notify gate and Event timer; public Combine is unchanged."""

import ctypes
import math
import os
import sys


class LaunchGate:
    def __init__(self, stream):
        self.stream = stream
        self.target = ctypes.c_void_p(stream.npu_stream)
        if not self.target.value:
            raise ValueError("Launch gate requires a live NPU stream")
        self.notify = ctypes.c_void_p()
        self.release = ctypes.c_void_p()
        self.pending = False
        self.closed = False
        self.timeout_ms = 180000
        self.acl = ctypes.CDLL("libascendcl.so")
        pointer = ctypes.c_void_p
        signatures = {
            "aclrtCreateNotify": [ctypes.POINTER(pointer), ctypes.c_uint64],
            "aclrtCreateStream": [ctypes.POINTER(pointer)],
            "aclrtWaitAndResetNotify": [pointer, pointer, ctypes.c_uint32],
            "aclrtRecordNotify": [pointer, pointer],
            "aclrtSynchronizeStreamWithTimeout": [pointer, ctypes.c_int32],
            "aclrtDestroyStream": [pointer],
            "aclrtDestroyNotify": [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.acl, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self.check(self.acl.aclrtCreateNotify(ctypes.byref(self.notify), 0), "create Notify")
        try:
            self.check(self.acl.aclrtCreateStream(ctypes.byref(self.release)), "create release stream")
        except BaseException:
            self.acl.aclrtDestroyNotify(self.notify)
            raise

    @staticmethod
    def check(status, operation):
        if status:
            raise RuntimeError(f"{operation}: ACL error {status}")

    @staticmethod
    def fail(error):
        # A partially submitted collective cannot safely unwind through Torch
        # cleanup. Let torchrun terminate peers instead of hanging on teardown.
        print(f"UBX launch gate failed: rank={os.environ.get('RANK', '?')} {error}", file=sys.stderr, flush=True)
        os._exit(1)

    def run(self, invoke):
        if self.closed or self.pending:
            raise RuntimeError("Launch gate is closed or already in use")
        if self.stream.npu_stream != self.target.value:
            raise RuntimeError("Launch gate stream changed")
        self.pending = True
        try:
            self.check(self.acl.aclrtWaitAndResetNotify(self.notify, self.target, self.timeout_ms), "arm Notify")
            result = invoke()
            # Flush torch_npu's Host task queue, including completion-event
            # submissions. This is not a device completion wait.
            if self.stream.npu_stream != self.target.value:
                raise RuntimeError("Launch gate stream changed")
            # HCCL and public Combine have both been submitted. This stream
            # must have no dependency on the measured stream it is releasing.
            self.check(self.acl.aclrtRecordNotify(self.notify, self.release), "release Notify")
            self.check(self.acl.aclrtSynchronizeStreamWithTimeout(self.target, self.timeout_ms), "Combine completion")
            self.check(self.acl.aclrtSynchronizeStreamWithTimeout(self.release, self.timeout_ms), "release completion")
        except BaseException as error:
            self.fail(error)
        self.pending = False
        return result

    def close(self):
        if self.closed:
            return
        if self.pending:
            self.fail("incomplete submission")
        try:
            self.check(self.acl.aclrtDestroyStream(self.release), "destroy release stream")
            self.check(self.acl.aclrtDestroyNotify(self.notify), "destroy Notify")
        except BaseException as error:
            self.fail(error)
        self.closed = True


class KernelEventTimer:
    """Bracket only the native launch reached through the public V2 adapter."""

    def __init__(self, native, stream):
        self.native = native
        self.target = ctypes.c_void_p(stream.npu_stream)
        self.start, self.end = ctypes.c_void_p(), ctypes.c_void_p()
        self.calls = 0
        self.acl = ctypes.CDLL("libascendcl.so")
        pointer = ctypes.c_void_p
        signatures = {
            "aclrtCreateEventWithFlag": [ctypes.POINTER(pointer), ctypes.c_uint32],
            "aclrtRecordEvent": [pointer, pointer],
            "aclrtEventElapsedTime": [ctypes.POINTER(ctypes.c_float), pointer, pointer],
            "aclrtDestroyEvent": [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.acl, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        # ACL_EVENT_TIME_LINE: the same flag as the former prepared benchmark.
        LaunchGate.check(self.acl.aclrtCreateEventWithFlag(ctypes.byref(self.start), 8), "create start Event")
        try:
            LaunchGate.check(self.acl.aclrtCreateEventWithFlag(ctypes.byref(self.end), 8), "create end Event")
        except BaseException:
            self.acl.aclrtDestroyEvent(self.start)
            raise

    def __enter__(self):
        self.original = self.native.combine_low_latency
        self.native.combine_low_latency = self.launch
        return self

    def launch(self, *args, **kwargs):
        try:
            stream = kwargs["stream"] if "stream" in kwargs else args[-1]
            if self.calls or stream != self.target.value:
                raise RuntimeError("Expected exactly one Combine launch on the measured stream")
            LaunchGate.check(self.acl.aclrtRecordEvent(self.start, self.target), "record start Event")
            result = self.original(*args, **kwargs)
            LaunchGate.check(self.acl.aclrtRecordEvent(self.end, self.target), "record end Event")
            self.calls = 1
            return result
        except BaseException as error:
            LaunchGate.fail(error)

    def elapsed_us(self):
        # The sampling loop waits for device completion before reading/reusing.
        if self.calls != 1:
            raise RuntimeError("The public Combine call did not reach the timed native launch")
        ms = ctypes.c_float()
        LaunchGate.check(self.acl.aclrtEventElapsedTime(ctypes.byref(ms), self.start, self.end), "Event elapsed time")
        value = ms.value * 1000
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid ACL Event duration")
        self.calls = 0
        return value

    def __exit__(self, exc_type, exc, traceback):
        self.native.combine_low_latency = self.original
        if exc is not None:
            LaunchGate.fail(exc)
        LaunchGate.check(self.acl.aclrtDestroyEvent(self.end), "destroy end Event")
        LaunchGate.check(self.acl.aclrtDestroyEvent(self.start), "destroy start Event")
