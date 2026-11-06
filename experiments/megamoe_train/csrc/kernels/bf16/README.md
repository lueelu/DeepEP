# MegaMoE BF16 training kernels (Ascend950)

This directory contains the BF16 forward and backward kernels used by
`deep_ep_experimental.megamoe_train.MegaMoeBF16Train`. See the
[module README](../../../README.md) for the supported API, wheel build and tests.
The implementation uses CATLASS and SHMEM; it does not require CATCCOS.

The root keeps the forward/backward entry sources and controller headers.
`catlass/` contains GEMM adapters and the L0C-to-UB copy extension; `epilogue/`
contains SwiGLU and saved-format operations; `layout/` defines memory layouts,
address views and context; `schedule/` contains readiness and scheduling helpers;
`routing/` contains routing initialization, weight gathering, combine and
unpermute. `common/` retains shared tiling data, constants, copy helpers and
`op_kernel/` utilities. Host-only tiling helpers live in `csrc/bf16/`.

## Computation and scheduling

```text
BF16 X -> routing/dispatch -> GMM1 -> SwiGLU + routing weight
       -> GMM2 -> combine/unpermute -> BF16 Y
```

The matrix products use FP32 accumulation and BF16 tensor outputs. SwiGLU uses
FP32 internal arithmetic with explicit BF16 boundaries; see the
[numerical contract](../../../docs/mega_moe_bf16_design.md#13-数学关系与舍入边界).

The ordinary BF16 forward GMM1 uses M192, with L1 `192 x 256 x 256` and L0
`192 x 256 x 64` tiles. GMM2 and backward retain M128. Forward dispatch copies
one row at a time with two UB slots. A Ready384 block covers two M192 tiles;
partial blocks use their actual valid row counts.

The default `adaptive-auto` policy chooses N stripes from actual rows, N tiles
and available AICs. For at most seven local experts and sufficient rows and
metadata capacity, eligible nonempty experts use Ready384. Small experts wait
for the whole expert. Backward BF16/MXFP8 saved modes use the existing M128 /
256-row progress policy where eligible; `none` retains recomputation. See the
[dispatch and synchronization design](../../../docs/mega_moe_bf16_design.md#37-前后向-dispatch-就绪与同步协议).

The module CMake file defines the supported device configuration. Build and
install a complete wheel using the root workflow. Device performance checks
use NPU Events; the operator API has no Timer parameters.

## State and memory ownership

The Python training wrapper returns `(y, counts)`. Native `_C.BF16.forward`
returns a `ForwardResult` containing `y`, `counts`, `saved_state` and
`dispatch_output`. Each outstanding forward owns its saved state and packed
dispatch input, retained by autograd for backward. Transient workspace is
managed inside the Host call and is not part of either result; stream recording
protects its asynchronous lifetime.

Use `TrainingSavedStateLayout` and the transient layout helpers in
`layout/mega_moe_layout.hpp` for sizing. Persistent state contains
packed X, input-typed BF16/FP32 routing weights, routing metadata, expert counts and optional
Gate/Up storage. Transient state contains forward payload and readiness flags;
control words are initialized for each invocation. The symmetric window belongs
to the root Session and is borrowed by this module.

Receive capacity, actual routed rows and per-expert alignment are different
quantities. Size from the selected shape and actual supported capacity rules;
input/output tensors, parameters, saved tensors and runtime memory are additional
to the symmetric window. Buffer formulas alone do not establish device support
for arbitrary EP sizes, subgroups or multiple nodes.

Canonical weights are `W1 [E, 2F, H]`, `W2 [E, H, F]`; supported alternative
layouts are `W1 [E, H, 2F]`, `W2 [E, F, H]`. The Host selects the matching
CATLASS layout from the tensor shape.

Only BF16 training is delivered here. W8A8 and UDMA-specific kernel
variants are not exposed by this module.
