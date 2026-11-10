# MegaMoE training component (Ascend950, experimental)

`deep_ep_experimental.megamoe_train.MegaMoeBF16Train` fuses routing, dispatch,
GMM1, SwiGLU, GMM2 and combine into one forward, with an autograd backward for
`x`, routing weights, `w1` and `w2`. Expert IDs and expert counts are not
differentiable. It targets Ascend950; other Ascend generations are unsupported.

| Document | Contents |
| --- | --- |
| [Quickstart (中文)](docs/BF16_QUICKSTART_ZH.md) | Minimal runnable example and the meaning of each parameter |
| [User guide (中文)](docs/BF16_USER_GUIDE_ZH.md) | Field-level contract, framework integration, capacity planning and limits |
| [Design (中文)](docs/mega_moe_bf16_design.md) | Architecture, forward/backward pipeline, numerical boundaries and synchronization |
| [Test entry](tests/bf16/README.md) · [Test spec](tests/bf16/TEST_SPEC.md) | How to run precision and performance checks, and how the results are judged |

## Build and installation

The commands below build the training component from a repository checkout. Environment
requirements (platform, CANN, Python, build tools, driver, SHMEM/CATLASS) are listed once in
the [root README](../../README.md#环境要求); packaging and multi-node installation details are
in the [build guide](../../docs/build.md).
Build in the target runtime image; transferred wheels must match its Python/Torch
ABI and system libraries, including glibc.

```bash
git submodule update --init third-party/shmem third-party/catlass
source /path/to/cann/set_env.sh
python -m pip install -r requirements-build.txt
python scripts/build_shmem.py
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export CATLASS_ROOT="$PWD/third-party/catlass"
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_BUILD_MEGAMOE_TRAIN=ON
python -m build --wheel --no-isolation
python -m pip install --force-reinstall --no-deps dist/*.whl
```

`SHMEM_ROOT` selects the build SDK. The complete wheel bundles that SDK's runtime
libraries under `deep_ep/.libs` and records their hashes. Installed training loads
those libraries through the root loader; only the matching CANN environment is
needed at run time. Do not preload another SHMEM SDK into the same process.
For a native import check, follow the [build skill](../../.agent/skills/megamoe_train/build-megamoe-train/SKILL.md#4-检查安装与原生扩展加载):
call the root `require_native()` loader before importing the training extension.
Normal `MegaMoeBF16Train` construction performs this bootstrap automatically.

The option defaults to OFF:

- Independent of `DEEPEP_BUILD_MEGAMOE` and `DEEPEP_BUILD_EP_GMM_FUSED`. Install a wheel, not an
  editable install. Training uses the root SHMEM runtime, does not load the inference kernel SO, and needs no
  CATCCOS.
- After changing Host or device sources, rebuild and reinstall the complete wheel: the Host
  extension, `libdeepep_megamoe_train_bf16.so` and the root runtime must come from one build, and
  every EP rank uses that build. BF16/FP32 routing weights share the public API and library
  names, but their typed tiling and payload layout require matching Host/device libraries;
  replacing only an old SO is unsupported. `deep_ep/_build_info.json` records the inputs.
- `DEEPEP_BF16_FORCE_EXPERT_READY=ON` (or `-DDEEPEP_BF16_FORCE_EXPERT_READY=ON`) builds the
  complete-expert dispatch fallback. OFF, the default, keeps M192/Ready384 forward dispatch and
  saved-mode row-ready backward. See the
  [user guide](docs/BF16_USER_GUIDE_ZH.md#73-实现上限与调度阈值) and the
  [design](docs/mega_moe_bf16_design.md#37-前后向-dispatch-就绪与同步协议).
- Empty `PYTHONPATH` entries, including a trailing colon, add the source directory and shadow the
  installed wheel.
- Agent runbook (中文): [build skill](../../.agent/skills/megamoe_train/build-megamoe-train/SKILL.md).

## Usage and lifetime

Set the current NPU and `DEEPEP_SHMEM_ENDPOINT=tcp://<rank-0-host>:<port>` before constructing the
module, and pass an existing ProcessGroup or `deep_ep` StoreGroup as `ep_group`; the caller owns that
group. All ranks use the same endpoint, capacity, shape, routing interpretation, save mode and call
order, and the same routing-weight dtype in each corresponding call.

```python
from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

op = MegaMoeBF16Train(ep_group, local_mem_size=1 << 30)  # per-rank SHMEM window, bytes
y, counts = op(x, topk_ids, topk_weights, w1, w2, max_recv_tokens,
               save_gmm1_output="bf16", swiglu_limit=0.0)
(y * grad_y).sum().backward()
op.close()  # collective, before destroying ep_group
```

A complete runnable script, including device selection, endpoint setup and input construction:
[quickstart §2 (中文)](docs/BF16_QUICKSTART_ZH.md#2-最小示例).

The constructor is `MegaMoeBF16Train(ep_group, local_mem_size, *, timeout=120,
debug_collectives=False)`. Existing calls retain the default. Local dtype, stream and lifetime
checks remain active. All ranks must agree on `debug_collectives` during initialization.
With the default `False`, forward/backward do not add a per-call dtype collective;
the framework must guarantee matching dtype and call order across EP ranks. `True` adds a
cross-rank dtype and forward/backward-stage check before each native call. It is a diagnostic
check, not a complete graph-identity or operation-order protocol, and its Host coordination
and waits are part of the measured call cost. If one rank exits or raises a local validation
error, the framework must terminate the distributed job rather than let other ranks continue.

The training Runtime owns a root Session and selects the MTE backend. Do not construct a second
DeepEP buffer or inference runtime concurrently in the same process.
Device qualification is single-node. ProcessGroup-based training integration,
multi-node transport, EP subgroups and larger EP sizes need separate
qualification before deployment; StoreGroup test coverage and the API's upper
rank bound are not guarantees for those integration paths or topologies.

- `x`: `[M,H]` BF16, `topk_ids`: `[M,T]` int32, `topk_weights`: `[M,T]` BF16 or FP32.
  `y`, `dx`, `dw1` and `dw2` remain BF16; `dtopk_weights` has the input p dtype.
  FP32 p is transported and used without an intermediate BF16 conversion.
  Tensor inputs are contiguous and reside on the construction NPU.
- Canonical `w1=[E,2F,H]`, `w2=[E,H,F]`; alternative layouts
  `w1=[E,H,2F]`, `w2=[E,F,H]` are also supported. Gradients use the input layouts.
- `max_recv_tokens` bounds received routed rows on each rank. The conservative
  bound is `EP*M*min(E,T)` for distinct IDs within each token; use actual route
  demand only when guaranteed. Duplicate IDs are not part of the supported
  routing contract.
- `local_mem_size` is the per-rank SHMEM communication window in bytes: at least 2 MiB, 2 MiB
  aligned, identical on every rank and fixed at construction. It covers only the window, not
  weights, saved activations or scratch. Plan the token payload, actual p width (2 or 4 bytes)
  and control area together, then round up to 2 MiB; an executable conservative planner is in
  [user guide §6.3](docs/BF16_USER_GUIDE_ZH.md#63-local_mem_size). An
  undersized window fails the Host layout check at call time, and enlarging it requires
  constructing a new operator.
- Calls must use the construction stream and be serialized in identical order
  across ranks. Keep weights unmodified until their corresponding backward.
  Saved state and packed dispatch output are independent per forward; scratch
  is transient. Multiple queued forwards can use BF16 or FP32 p and retain their own dtype;
  their corresponding backwards must follow a consistent order across ranks.
  Cross-stream/concurrent calls and higher-order derivatives
  are unsupported. `retain_graph=True` requires keeping the operator open for
  every subsequent backward.
- `close()` synchronizes the launch stream and releases the root Session
  collectively. Callers must complete all required backwards or discard unused
  graphs before close; the operator does not track pending graphs. Garbage
  collection never performs collective SHMEM cleanup. After a device/collective
  failure restart the job.
- Public `forward` retains data required for backward. Device runners use the
  same public interface and validate final outputs, counts and gradients.

## Gate/Up save modes

`save_gmm1_output` selects how the GMM1 output is kept for backward. GMM1 emits Gate/Up, the two
linear results that SwiGLU consumes; the saved state stores those rows, not the SwiGLU output H.

| Mode | Gate/Up used by backward |
| --- | --- |
| `bf16` (default) | Saved BF16 values |
| `none` | Recomputed GMM1 |
| `mxfp8` | Saved MXFP8, plus three clip-mask bitplanes when clipping is enabled |
| `auto` | MXFP8 without clipping, BF16 with clipping |

All four modes accept BF16 or FP32 p, independently of how Gate/Up is saved; p is never
compressed to MXFP8. The typical model suite uses `bf16` saving, while the dtype suite
also covers `none`, `mxfp8` and `auto` configurations. `mxfp8` and `auto` remain experimental:
MXFP8 compression introduces its own error, so a reconstructed H is not bit-exact with the
uncompressed forward. A case being listed does not establish precision qualification.
Coverage and acceptance rules are stated in
[test spec §1](tests/bf16/TEST_SPEC.md#1-范围与模型参数).

`swiglu_limit<=0` disables clipping; a positive finite limit L clips Gate to the upper bound L and
Up to `[-L,L]`. MXFP8 is the saved activation format; matrix computation stays BF16. Scheduling
policy for GMM1, SwiGLU decoupling and backward dispatch splitting:
[design §3.7](docs/mega_moe_bf16_design.md#37-前后向-dispatch-就绪与同步协议).

## Tests

The published [case manifest](tests/bf16/cases.json) contains 42 cases: 12 typical BF16
model-dimension cases (`priority` / `model`), 24 p-dtype/save-mode/clipping cases (`dtype`),
and 6 FP32 gather-window boundary cases (`gather`). `all` selects all 42. Other data profiles
and historical matrices are not discovered automatically.

| Model dimensions | H | F / GMM1 N | TopK | Experts/rank | Clip | Route-weight scale | M/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Flash | 5120 | 2304 / 4608 | 6 | 4 | 10 | 1.5 | 512 / 2048 / 4096 / 8192 |
| Pro | 7168 | 3072 / 6144 | 6 | 4 | 10 | 2.5 | 512 / 2048 / 4096 / 8192 |
| GLM | 6144 | 2048 / 4096 | 8 | 8 | 0 | 2.5 | 512 / 2048 / 4096 / 8192 |

The 12 model cases use balanced routing, BF16 p, BF16 saving and seed 20260825. These are synthetic
operator tests with reduced expert counts, not full-model or convergence validation.
M means tokens entering the operator per rank; EP alone does not split M.
The upstream dy is fixed-seed random input shared by all comparison paths;
y.backward(dy) computes all four input gradients.
See the [test spec (中文)](tests/bf16/TEST_SPEC.md) for the rationale and known limits.

There is one entry, `tests/bf16/test_mega_moe.py`, with these helpers:

- `cpu_reference.py`: independent Torch CPU FP64 forward and autograd truth.
- `npu_baseline.py`: FP32-boundary NPU BigOp and the auxiliary numerical-contract reference.
- `atk_compare.py`: the private ATK CV double-benchmark comparator.
- `precision_metrics.py`: exact counts and mixed-tolerance reports.
- `mxfp8_diagnostics.py`: optional diagnosis for explicitly configured inputs;
  the dtype suite independently includes MXFP8 save-mode cases.

### Run the typical cases

Precision needs `atk` and `atk-private-precision` in the same Python as the wheel; performance
needs neither.

```bash
# Precision on one case: forward, all four gradients and counts.
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite model --case-id flash_m512 --skip performance \
  --shmem-ip-port tcp://127.0.0.1:29862

# Performance on one case, without ATK or CPU FP64 reference computation.
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite model --case-id pro_m2048 --skip precision \
  --shmem-ip-port tcp://127.0.0.1:29862
```

### Check routing-weight dtypes

For an explicit shape, `--topk-weight-dtype bf16/fp32` defaults to BF16. FP32 defaults to
`--topk-weight-values full-precision`; select `bf16-rounded` to supply the same values as
BF16 p widened with `.float()`. Other seeded inputs remain unchanged. The CPU FP64 truth,
BigOp and fused operator consume the selected p values; the reference version is
`fp32-boundaries-v3-topk-dtype`. Suite settings come from the manifest.

```bash
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite dtype --skip performance --shmem-ip-port tcp://127.0.0.1:29862
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite gather --skip performance --shmem-ip-port tcp://127.0.0.1:29862
```

These are executable test configurations, not claims that every listed mode, shape or EP
has passed on the installed wheel. Provision sufficient host RAM and device memory for the
FP64 reference and retained gradients before selecting a suite.

Case selection, custom shapes, metric definitions, report fields and known issues:
[runbook](tests/bf16/README.md) · [test spec (中文)](tests/bf16/TEST_SPEC.md).
Agent runbooks (中文): [ATK skill](../../.agent/skills/megamoe_train/test-megamoe-train-atk/SKILL.md) ·
[performance skill](../../.agent/skills/megamoe_train/test-megamoe-train-perf/SKILL.md).
