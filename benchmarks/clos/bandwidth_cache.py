"""Per-rank cache eviction, outside the measured public API invocation."""

L2_CACHE_FLUSH_BYTES = 256_000_000
L2_FLUSH_RANGE = "deepep_benchmark_l2_flush"
PROFILE_SAMPLE_RANGE = "deepep_benchmark_sample/"
PROFILE_GUARD_ITERS = 2


class ProfileSamples:
    """Device event boundaries, linked to explicit sample IDs by profiler flows.

    Keep events alive until export finishes. Host annotation timestamps must
    never be used as device execution boundaries (their clocks can differ).
    """

    def __init__(self, torch):
        self.torch = torch
        self.events = []

    def __call__(self, sample, invoke):
        def mark(edge):
            event = self.torch.npu.Event()
            self.events.append(event)
            with self.torch.autograd.profiler.record_function(f"{PROFILE_SAMPLE_RANGE}{sample}/{edge}"):
                event.record()

        mark("start")
        invoke()  # includes the public API's completion wait on the caller stream
        mark("end")


class L2CacheFlush:
    def __init__(self, torch, buffer, enabled):
        self.torch = torch
        self.buffer = buffer
        self.enabled = enabled
        if enabled:
            self.stream = torch.npu.Stream()
            with torch.npu.stream(self.stream):
                self.cache = torch.empty(L2_CACHE_FLUSH_BYTES // 4, dtype=torch.int32, device="npu")

    def __call__(self):
        if not self.enabled:
            return
        # Finish the preceding invocation before eviction. The host wait also
        # separates calls; the parser identifies zero_ using profiler flows,
        # not by treating this Host annotation as a device execution interval.
        self.stream.wait_stream(self.torch.npu.current_stream())
        self.stream.wait_stream(self.buffer.get_comm_stream())
        self.stream.synchronize()
        with self.torch.autograd.profiler.record_function(L2_FLUSH_RANGE):
            with self.torch.npu.stream(self.stream):
                self.cache.zero_()
            self.stream.synchronize()

    def wrap(self, fn):
        def invoke():
            self()
            return fn()

        return invoke
