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

`benchmarks/clos/bench_v2_bandwidth.py` preserves the migrated one-command EP8/EP16
benchmark and stage accounting. The original POD benchmarks stay in their
original paths. Separate runs are needed to qualify each backend and topology;
compilation or CPU tests alone do not establish NPU correctness/performance.
