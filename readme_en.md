<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# DeepEP

---

[简体中文](README.md) | English

**DeepEP on Ascend** (`Ascend/DeepEP`) is an expert-parallel (EP) communication library for mixture-of-experts (MoE) inference on Ascend NPUs. It provides high-throughput **prefill** and low-latency **decode** communication, together with experimental **MegaMoE** fused inference.

## News

---

**DeepEP V2 on Ascend.** Prefill and decode share the `deep_ep.ElasticBuffer` interface and its `dispatch()` / `combine()` operations.

### New features

- **High-throughput prefill:** URMA dispatch and combine, integrated notify, and server-level token deduplication, supporting EP32/64/128/256; see the [Elastic guide](docs/elastic.md) for the detailed support matrix.
- **Low-latency decode:** small-batch dispatch and combine through the same V2 interface for EP64/128.
- **MegaMoE prefill (experimental):** W4A8 fused communication and expert computation for EP32/64/128.
- **MegaMoE decode (experimental):** small-batch W4A8 fused inference for EP64/128, using the same MegaMoE forward entry as prefill.

### Architecture

DeepEP connects inference frameworks to Ascend communication kernels through V2. MegaMoE integrates token communication, expert computation, activation, and quantization into a fused forward workflow.

<p align="center">
  <a href="figures/architecture_en.svg">
    <img src="figures/architecture_en.svg" alt="DeepEP software architecture" width="800">
  </a>
</p>

```text
DeepEP/
├── deep_ep/
│   ├── buffers/             # Python interfaces
│   └── utils/               # Events and shared utilities
├── csrc/
│   ├── bindings/            # Python/C++ bindings
│   ├── runtime/             # Communication resources and lifecycle
│   ├── ops/                 # Host validation and kernel launch
│   └── kernels/             # Ascend C communication kernels
├── experiments/
│   ├── megamoe/             # MegaMoE fused inference
│   └── ep_gmm_fused/        # Fused communication and grouped matmul
├── cmake/                   # Build and installation
├── third-party/             # Dependencies
├── configs/                 # Deployment and execution configuration
├── tests/                   # Correctness and regression tests
├── benchmarks/              # Performance benchmarks
├── examples/                # Integration examples
├── docs/                    # Detailed documentation
└── figures/                 # Architecture and performance figures
```

See the [architecture guide](docs/architecture.md) for module responsibilities.

## Quick start

---

### Requirements

| Component | Requirement |
| --- | --- |
| Platform | Linux, Ascend950, CANN 9.2.0(Public availability is currently planned for around October 15 2026) |
| Python | ≥ 3.11 |
| Build tools | C++17, CMake ≥ 3.20, Make or Ninja, Ascend C/Bisheng |
| Framework | PyTorch and `torch_npu` |
| HDK and Firmware | Huawei's Q3 commercial HDK with a planned release date of October 15 2026 |
| Communication | SHMEM pinned in [dependencies.lock.json](dependencies.lock.json); URMA runtime |
| MegaMoE | CATLASS; dependency setup in the [MegaMoE guide](experiments/megamoe/README.md) |

### Install SHMEM dependency

Prepare the SHMEM SDK using the [build guide](docs/build.md), then load the CANN and SHMEM environment:

```bash
source /path/to/ascend-toolkit/set_env.sh
export SHMEM_ROOT=/path/to/shmem-sdk
source "$SHMEM_ROOT/set_env.sh"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

### Development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/elastic/test_dispatch.py tests/elastic/test_combine.py -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and the [test guide](tests/README.md) for NPU test commands.

### Installation

```bash
git clone https://github.com/lueelu/DeepEP.git
cd DeepEP
export DEEPEP_NPU_ARCH=Ascend950
python -m pip install -r requirements-build.txt
python -m pip install -v --no-build-isolation --no-deps .
```

The package name is `ascend-deepep`; the Python import is `deep_ep`. See the [build guide](docs/build.md) for wheel packaging and multi-node installation.

### Install MegaMoE

With matching Torch/torch_npu and CANN 9.2.0, run:

```bash
bash experiments/megamoe/scripts/build.sh
```

The script prepares dependencies and builds and installs a wheel with MegaMoE and timer support. Subsequent builds reuse the validated SHMEM SDK. Direct `pip wheel` builds keep MegaMoE disabled by default. See the [MegaMoE guide](experiments/megamoe/README.md) for multi-node configuration and validation. Device compilation, accuracy and performance qualification remain outstanding.

## Interfaces and examples

---

| Workload | Interface |
| --- | --- |
| DeepEP prefill / decode | `deep_ep.ElasticBuffer.dispatch()` / `combine()` |
| MegaMoE prefill / decode (experimental) | `deep_ep_experimental.megamoe.fp8_fp4_mega_moe()` |

### Buffer initialization

Prefill and decode share `ElasticBuffer.dispatch()` / `combine()`. For supported configurations, `num_max_tokens_per_rank ≤ 256` uses low latency; other configurations use high throughput. This parameter is the per-rank token capacity, not the current token count.

The BF16 example below assumes that the NPU device, SHMEM endpoint and EP process group `ep_group` are configured. The model supplies `x [T,H]` (BF16), `topk_idx [T,K]` (INT64) and `topk_weights [T,K]` (FP32) on the current NPU. All ranks use the same T/H/K, with `T ≤ max_tokens_per_rank` and an expert count divisible by the EP size. High throughput supports `0 < H ≤ 7168`: H must be divisible by 16 for BF16 or 128 for FP8, with 1≤K≤16.

Low latency uses an 8 GiB workspace; for high throughput, the buffer estimates the capacity automatically.

```python
from deep_ep import ElasticBuffer

max_tokens_per_rank = 256
num_experts = 384
hidden = x.shape[1]
num_topk = topk_idx.shape[1]

buffer = ElasticBuffer(
    group=ep_group,
    num_bytes=(8 << 30) if max_tokens_per_rank <= 256 else None,
    num_max_tokens_per_rank=max_tokens_per_rank,
    hidden=hidden,
    num_topk=num_topk,
    explicitly_destroy=True,
)
```

Use the same configuration on all ranks and reuse the buffer across iterations. Inference uses only the forward steps; the backward example is for high-throughput training. See the [Elastic guide](docs/elastic.md) for supported configurations.

### Dispatch/Combine example

Forward runs Dispatch → expert computation and weighting → Combine. Backward runs Dispatch → expert backward → Combine. Pass the handle returned by the corresponding forward Dispatch to Combine and both backward calls.

The four example functions below wrap communication calls and waits. The model provides expert computation and gradient computation. Communication inputs must be contiguous and detached.

```python
def dispatch_forward(x, topk_idx, topk_weights):
    recv_x, _, recv_weights, handle, event = buffer.dispatch(
        x.detach().contiguous(),
        topk_idx=topk_idx.contiguous(),
        topk_weights=topk_weights.detach().contiguous(),
        num_experts=num_experts,
        do_expand=True,
    )
    event.current_stream_wait()
    return recv_x, recv_weights, handle


def dispatch_backward(grad_recv_x, grad_recv_weights, handle):
    grad_x, grad_weights, event = buffer.combine(
        grad_recv_x.detach().contiguous(),
        handle,
        topk_weights=grad_recv_weights.detach().contiguous(),
    )
    event.current_stream_wait()
    return grad_x, grad_weights


def combine_forward(expert_out, handle):
    out, _, event = buffer.combine(expert_out.detach().contiguous(), handle)
    event.current_stream_wait()
    return out


def combine_backward(grad_out, handle):
    grad_expert_out, _, _, _, event = buffer.dispatch(grad_out.detach().contiguous(), handle=handle)
    event.current_stream_wait()
    return grad_expert_out
```

`local_experts_forward/backward` are supplied by the model, not DeepEP APIs. Forward computes the expert FFN and applies `recv_weights` once, returning BF16 outputs in the received row order and an `expert_ctx` for backward. Backward computes expert parameter gradients and returns BF16 input gradients and FP32 weight gradients with the same shapes as `recv_x` and `recv_weights`, respectively.

```python
# Forward: training and inference use the same calls.
recv_x, recv_weights, handle = dispatch_forward(x, topk_idx, topk_weights)
expert_out, expert_ctx = local_experts_forward(recv_x, recv_weights, handle)
out = combine_forward(expert_out, handle)
```

During training, call these steps in the model's backward pass:

```python
# The model supplies grad_out during backward, in BF16 with the same shape as out.
grad_expert_out = combine_backward(grad_out, handle)
grad_recv_x, grad_recv_weights = local_experts_backward(grad_expert_out, expert_ctx)
grad_x, grad_topk_weights = dispatch_backward(grad_recv_x, grad_recv_weights, handle)
```

The model feeds `grad_x` and `grad_topk_weights` back into the input network and router, respectively. Inference uses only the forward steps.

Every rank must participate in communication in the same order, including ranks with no received rows. For training, keep each handle until its backward pass finishes. After all iterations, call `buffer.destroy()` in the same order on all ranks, before releasing the EP group. See the [Elastic guide](docs/elastic.md) and [API guide](docs/api.md) for FP8, asynchronous calls and handle reuse.

### MegaMoE example

MegaMoE fuses token dispatch, two expert matrix multiplications, SwiGLU, quantization, and combine into one forward call. Prefill and decode use the same interface.

The integration snippet below uses FP8 inputs, FP4 weights, and one shared expert per rank. Launch with `torchrun` and set the same `DEEPEP_SHMEM_ENDPOINT=tcp://<master-IPv4>:<unused-port>` on every rank, using a port distinct from `MASTER_PORT`.
The model supplies `x_fp8`, `x_sf`, `topk_idx`, `topk_weights`, and raw ND `(weight, scale)` pairs for routed and shared experts on the current NPU. See the [MegaMoE input and weight layouts](experiments/megamoe/README.md) for shapes and dtypes.

```python
import os
import torch
import torch_npu  # Register the NPU backend.
from deep_ep_experimental.megamoe import (
    StoreGroup, SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe,
)

torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
group = StoreGroup.from_env()
try:
    # Initialize once: 384 routed experts and one shared expert per rank.
    buffer = SymmBuffer(
        group, num_experts=384, num_max_tokens_per_rank=4096,
        num_topk=6, hidden=7168, intermediate_hidden=3072,
        num_shared_experts=1, mma_type="fp8xfp4",
        ranks_per_node=int(os.environ["LOCAL_WORLD_SIZE"]),
    )
    w1, w2 = transform_weights_for_mega_moe(l1_weights, l2_weights)
    shared_w1, shared_w2 = transform_weights_for_mega_moe(
        shared_l1_weights, shared_l2_weights, shared=True,
    )

    # Fill inputs before each call; all ranks use the same num_tokens, within buffer capacity.
    num_tokens = x_fp8.shape[0]
    buffer.x[:num_tokens].copy_(x_fp8)                 # FP8 E4M3 [B, H]
    buffer.x_sf[:num_tokens].copy_(x_sf)              # E8M0 [B, H / 32]
    buffer.topk_idx[:num_tokens].copy_(topk_idx)      # int32 [B, K]
    buffer.topk_weights[:num_tokens].copy_(topk_weights)  # FP32 [B, K]
    buffer.validate_inputs(num_tokens)  # Validate and prepare outside the timed region.
    buffer.prepare(num_tokens)
    y = torch.empty((num_tokens, 7168), dtype=torch.bfloat16, device="npu")
    fp8_fp4_mega_moe(
        y, w1, w2, buffer,
        shared_l1_weights=shared_w1, shared_l2_weights=shared_w2,
    )  # Writes into y and returns None.
    torch.npu.synchronize()

    # Reuse the buffer and prepared weights; after all calls, release in the same order on every rank.
    group.barrier("before SHMEM teardown")
    buffer.destroy()
    group.close()
except BaseException as error:
    group.abort(f"MegaMoE: {type(error).__name__}: {error}")
    raise
```

For routed experts only, set `num_shared_experts=0` and omit shared-weight conversion and arguments. Keep the returned weight pairs intact and reuse them.
See [example.py](experiments/megamoe/python/deep_ep_experimental/megamoe/example.py) for input generation and a runnable multi-process example, and the [MegaMoE guide](experiments/megamoe/README.md) for the full interface contract.

Tensor layouts, event synchronization, and buffer lifecycle are described in the [API guide](docs/api.md) and [Elastic guide](docs/elastic.md).

### Environment variables

| Variable | Purpose |
| --- | --- |
| `DEEPEP_NPU_ARCH` | Build target: `Ascend950` |
| `SHMEM_ROOT` | SHMEM SDK directory |
| `DEEPEP_SHMEM_ENDPOINT` | Shared initialization endpoint for the EP job |
| `ASCEND_RT_VISIBLE_DEVICES` | NPU devices assigned to the job |

## Network configurations

---

Each EP group uses a shared SHMEM initialization endpoint and an explicit rank-to-device mapping. The current implementation groups consecutive sets of eight ranks into logical servers for server-level deduplication; a logical server need not match a physical host.

## Experimental branches

---

### MegaMoE

MegaMoE combines dispatch, expert computation, activation/quantization, and combine into W4A8 fused inference for prefill and decode. See [experiments/megamoe/](experiments/megamoe/README.md) for configuration and usage.

## Acknowledgement

---

We thank [DeepSeek DeepEP](https://github.com/deepseek-ai/DeepEP) for the upstream EP interface and workflow reference, and the Ascend communication and compute library contributors supporting this project.

## License

---

Project material for which Lu Lu holds the relevant rights is licensed under the [BSD 2-Clause License](LICENSE). Other source material retains its own copyright notices and license texts.
# Machine-specific backends

This branch adds the Clos V1/V2 implementation alongside the existing POD backend,
using the same pinned CANN SHMEM dependency and build workflow. Buffer constructors
accept `machine_type="auto"`, `"pod"`, `"clos_only"`, or `"card_4pmesh"`; explicit
selection is available when reliable rootinfo is absent. The machine selects the
communication engines: Clos uses MTE + UDMA, CARD_4PMESH uses MTE, and POD keeps its
existing configuration. Public constructors do not accept `transport`; remove it
from older calls. `ASCEND_DEEPEP_TRANSPORT` does not override this policy.
See [backend selection and compatibility](docs/machine_backends.md).
