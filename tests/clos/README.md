# Clos backend regression tests

The imported tests retain the `deepep` c090403 CPU references, routes, tolerances
and kernel expectations. Backend tests use `deep_ep.backends.clos` explicitly so that
running them on a POD development machine does not silently test the POD path.
Public selection, naming, event compatibility and EP16 sizing tests are in
`tests/api/test_machine_selection.py` and `test_clos_integration.py`.

CPU API and golden checks (matching Torch/torch_npu required, no NPU launch):

```bash
python -m pytest tests/clos/source/test_public_api.py \
    tests/clos/source/test_v2_elastic_api.py \
    tests/clos/source/test_shmem_bootstrap.py \
    tests/clos/source/test_stage_metrics.py
```

The old source repository's build-identity test is superseded by
`tests/runtime/test_build.py`, `test_loader.py` and the shared SDK manifest checks.
NPU precision scripts remain manual commands; do not use a busy node:

```bash
python tests/clos/source/test_v1_intranode.py --ep-size 8 --transport udma \
    --num-tokens 129 --hidden 6144 --dtype bf16 --num-aiv 32
python tests/clos/source/test_v2_elastic_v1_kernel.py --mode expand --ep-size 8 --num-aiv 32
```

To check the installed public interface and automatic selection on a Clos node,
first run the CPU-only topology check, then run each API separately on idle cards:

```bash
python -I scripts/check_machine.py --physical-devices 0 1 2 3 4 5 6 7 --expect clos_only
python tests/clos/source/test_v1_v2_shared_backend.py --api v1 --public-auto \
    --ep-size 8 --num-tokens 129 --hidden 6144 --num-topk 8 \
    --experts-per-rank 32 --num-aiv 32 --check-events
python tests/clos/source/test_v1_v2_shared_backend.py --api v2 --public-auto \
    --ep-size 8 --num-tokens 129 --hidden 6144 --num-topk 8 \
    --experts-per-rank 32 --num-aiv 32 --check-events
```

`--expect clos_only` checks the detection result; it does not force the backend.
`--public-auto` uses the installed public Buffer/ElasticBuffer with the default
automatic selection and no transport override. It rejects other topologies
before Buffer construction because this test uses the Clos CPU golden.
Each rank prints `MACHINE_SELECTION` with the actual Buffer selection and engine.
`PUBLIC_PRECISION_PASSED` is printed only after BF16 fresh/cached dispatch,
combine with/without weights and bias, event dependencies, retained output after
destroy, and process-group cleanup succeed. Require all eight rank markers and
a zero launcher exit code. Keep an external job timeout when running these tests.

`benchmarks/clos/bench_v2_bandwidth.py` preserves the migrated one-command EP8/EP16
benchmark and stage accounting. The original POD benchmarks stay in their
original paths. Separate runs are needed to qualify each backend and topology;
compilation or CPU tests alone do not establish NPU correctness/performance.
