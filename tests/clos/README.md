# Clos basic validation

The Clos test directory keeps two files:

| File | Purpose |
| --- | --- |
| `source/test_v1_v2_shared_backend.py` | Basic V1/V2 public-interface precision and event checks |
| `common.py` | Shared independent CPU golden and precision helpers used by the basic checks and both bandwidth scripts; no test launcher or profiling |

Standalone stage tests, scheduling models, CPU CANN stubs, and specialized
regression scripts are not included. Shared public-API and topology tests remain
in `tests/api/`. No CPU stub result is a substitute for device precision checks.

## Machine/topology detection

Check the installed package without launching communication kernels:

```bash
python -I scripts/check_machine.py --physical-devices 0 1 2 3 4 5 6 7 --expect clos_only
```

`--expect` checks the detected topology; it does not force a backend. Use `pod`
or `card_4pmesh` for those machines. This checks classification, not link health
or SHMEM initialization.

## Basic functional validation

Install the wheel built from the revision being checked, load the CANN/SHMEM
environment, and run on idle cards. On a Clos machine:

```bash
for api in v1 v2; do
    timeout -k 30s 900s python tests/clos/source/test_v1_v2_shared_backend.py \
        --api "$api" --public-auto --auto-policy \
        --ep-size 8 --num-tokens 129 --hidden 6144 --num-topk 8 \
        --experts-per-rank 32 --num-aiv 32 --check-events
done
```

This checks BF16 fresh/cached dispatch, payload and route weights, combine
with/without weights and bias, event dependencies, and retained output after
destroy against the CPU golden. Require `PUBLIC_PRECISION_PASSED` for every
rank and a zero launcher exit code. `--auto-policy` uses the default scheduler
while preserving the requested AIV count.

For forced Clos validation on a development machine, replace `--public-auto`
with `--public-clos`. This does not qualify that machine's automatic Clos
selection. The basic launcher is single-node; `--ep-size` is its local process
count. Use the bandwidth entrypoints below for multi-node precision checks.

## Additional dtype/layout coverage and profiling

The existing `benchmarks/clos/bench_v1_bandwidth.py` and
`benchmarks/clos/bench_v2_bandwidth.py` check precision before measuring
fresh/cached dispatch and weighted/unweighted combine. Their defaults cover
BF16/FP8 and expanded/non-expanded layouts using the default scheduling policy.
They retain configurable shapes, routing, AIV count and multi-node EP32/64
launching; see [backend usage](../../docs/machine_backends.md).

Both scripts depend on `tests/clos/common.py` for precision checks and share
`benchmarks/clos/profiling.py` for measurement. The basic test has no profiling
options. Keep these helpers with the benchmark checkout.
The benchmarks load the installed package and record its library identity, while
the CPU reference comes from the checkout. Do not put an unbuilt checkout ahead
of the wheel in `PYTHONPATH`.

Passing these basic cases does not establish graph, subgroup, stress, or every
EP/shape combination. Qualify additional scenarios before relying on them.
