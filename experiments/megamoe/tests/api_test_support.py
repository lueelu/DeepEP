# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Execute public wrappers using CPU storage doubles; this does not simulate the NPU kernel."""

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api


class Device(str):
    @property
    def type(self):
        return self.split(":")[0]


DTYPES = dict(
    uint8=np.uint8,
    int8=np.int8,
    int32=np.int32,
    int64=np.int64,
    float32=np.float32,
    bfloat16=np.float32,
    float8_e4m3fn=np.uint8,
    float8_e8m0fnu=np.uint8,
)


class Tensor:
    def __init__(self, data, dtype, device="npu:0"):
        self.data, self.dtype, self.device = np.asarray(data, dtype=DTYPES[dtype]), dtype, Device(device)
        self.requires_grad = False
        self.streams = []
        self.npu_format = 2
        self._version = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, index):
        return Tensor(self.data[index], self.dtype, self.device)

    def view(self, dtype):
        return Tensor(self.data.view(DTYPES[dtype]), dtype, self.device)

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape), self.dtype, self.device)

    def transpose(self, first, second):
        return Tensor(self.data.swapaxes(first, second), self.dtype, self.device)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data), self.dtype, self.device)

    def clone(self):
        return Tensor(self.data.copy(), self.dtype, self.device)

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def untyped_storage(self):
        base = self.data
        while isinstance(base.base, np.ndarray):
            base = base.base
        return SimpleNamespace(data_ptr=lambda: base.ctypes.data)

    def record_stream(self, stream):
        self.streams.append(stream)

    def cpu(self):
        return self

    def tolist(self):
        return self.data.tolist()

    def add_(self, other):
        self.data += other.data
        self._version += 1
        return self


@pytest.fixture(name="backend")
def quantized_backend(monkeypatch):
    events = []

    class Group:
        def size(self):
            return 2

    def gather(output, value, **kwargs):
        output[:] = [value] * 2

    dist = SimpleNamespace(ProcessGroup=Group, get_world_size=lambda _: 2, all_gather_object=gather)
    torch = SimpleNamespace(
        **{name: name for name in DTYPES},
        device=Device,
        empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=DTYPES[dtype]), dtype, device),
        zeros=lambda shape, dtype, device: Tensor(np.zeros(shape), dtype, device),
        empty_like=lambda t, dtype: Tensor(np.empty(t.shape, dtype=DTYPES[dtype]), dtype, t.device),
        distributed=dist,
        inference_mode=lambda _: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    def format_cast(t, acl_format, *, customize_dtype, input_dtype):
        # 检查实际传入的协议，不能把所有 format_cast 参数都静默忽略。
        assert t.dtype == "uint8" and acl_format == 29
        assert customize_dtype == "float8_e4m3fn" and input_dtype == "float4_e2m1fn_x2"
        events.append(("format_cast", t))
        t = Tensor(t.data.copy(), t.dtype, t.device)
        t.npu_format = 50  # FP4 打包成 uint8 后，输出码与请求的29不同。
        return t

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda t: t.npu_format, npu_format_cast=format_cast, float4_e2m1fn_x2="float4_e2m1fn_x2"
        ),
    )
    stream = SimpleNamespace(wait_stream=lambda _: events.append("wait"))
    npu = SimpleNamespace(
        current_device=lambda: 0, current_stream=lambda _: stream, stream=lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(api, "npu_api", lambda: npu)
    monkeypatch.setattr(api, "require_native", lambda _: SimpleNamespace(megamoe_workspace_address=lambda _: 4096))

    class Runtime:
        def __init__(self, group, capacity, timeout, **options):
            self.device, self.stream, self.native = 0, stream, SimpleNamespace(capacity=capacity)
            self.closed = False

        def ensure_alive(self):
            if self.closed:
                raise RuntimeError("Buffer runtime has been destroyed")

        def barrier(self):
            events.append("barrier")

        def close(self):
            events.append("close")
            self.closed = True

    monkeypatch.setattr(api, "_MegaRuntime", Runtime)

    class Plan:
        def __init__(self, example, *args):
            self.bs, self.workspace_bytes = example.shape[0], 4096
            events.append(("prepare", self.bs))

        def forward(self, *args):
            events.append(("launch", args))
            output = args[-1]
            output.data.fill(3)
            return output, Tensor([2, 3], "int32")

    ext = SimpleNamespace(
        buffer_size=lambda *args: 2 << 20,
        Plan=Plan,
        ffts_config=lambda: 123,
        timer_enabled=True,
        timer_numel=64,
        small_batch_max_tokens=256,
    )
    monkeypatch.setattr(api, "_extension", lambda: ext)
    buffer = api.SymmBuffer(Group(), 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    yield SimpleNamespace(buffer=buffer, events=events, ext=ext, dist=dist)
    buffer.destroy()


def weights():
    prepared = (
        (Tensor(np.zeros((2, 512, 512)), "uint8"), Tensor(np.full((2, 512, 16, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 1024, 128)), "uint8"), Tensor(np.full((2, 1024, 4, 2), 127), "float8_e8m0fnu")),
    )
    for weight, _ in prepared:
        weight.npu_format = 50
    return prepared
