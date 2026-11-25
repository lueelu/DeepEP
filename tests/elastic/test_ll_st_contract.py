# Copyright (c) 2026, Lu Lu
# Modified by ryan_li 2026

"""CPU checks for the LL device ST launcher, reference and measurement contract."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("ll_st", Path(__file__).with_name("test_ep_low_latency.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def test_launch_identities():
    args = st.make_parser().parse_args(["--num-nodes", "2", "--node-rank", "1"])
    assert st.launch_config(args, {}) == (16, 8, None)
    assert st.launch_config(args, {"WORLD_SIZE": "4", "RANK": "2"}) == (32, 16, None)
    assert st.launch_config(args, {"WORLD_SIZE": "16", "RANK": "9", "LOCAL_RANK": "1"}) == (16, 9, 1)
    with pytest.raises(ValueError, match="torchrun"):
        st.launch_config(args, {"LOCAL_RANK": "0"})


@pytest.mark.parametrize(
    "world,argv",
    [
        (8, []),
        (16, ["--num-max-tokens-per-rank", "257"]),
        (16, ["--num-tokens", "32", "--num-max-tokens-per-rank", "16"]),
        (16, ["--num-sms", "64"]),
        (16, ["--num-qps", "2"]),
        (16, ["--num-bytes", str(2 << 20)]),
        (16, ["--num-bytes", str(10 << 30)]),
    ],
)
def test_reject_ht_cases(world, argv):
    with pytest.raises(ValueError):
        st.validate_args(st.make_parser().parse_args(argv), world)


def test_ll_capacity_boundary_and_defaults():
    args = st.make_parser().parse_args(["--num-tokens", "8", "--num-max-tokens-per-rank", "255"])
    assert st.validate_args(args, 16) == 255
    assert not args.skip_check and not args.skip_perf_test and args.repeat_combine >= 2
    assert st.validate_args(st.make_parser().parse_args([]), 16) == 16


def test_unique_routes_and_expanded_counts():
    routes = st.make_routes(16, 8, 8, 128, 42)
    assert all(len(set(row)) == 8 for rows in routes for row in rows)
    assert sum(len(rows) for rank in range(16) for rows in st.expert_entries(routes, 128, 16, rank)) == 16 * 8 * 8


def test_kernel_csv_contract(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text(
        "Op Name,Task Duration(us),Start Time(us)\ncombine_server_dedup_kernel,4,20\ncombine_server_dedup_kernel,2,10\ncombine_server_dedup_kernel_v1,99,1\nHcomAllGather,99,1\n"
    )
    assert st.kernel_durations(tmp_path, "combine_server_dedup_kernel", 2) == [2, 4]
    with pytest.raises(ValueError, match="expected 3"):
        st.kernel_durations(tmp_path, "combine_server_dedup_kernel", 3)
    path.write_text("Name,Duration(us)\ncombine_server_dedup_kernel,nan\n")
    with pytest.raises(ValueError, match="Invalid kernel duration"):
        st.kernel_durations(tmp_path, "combine_server_dedup_kernel", 1)


@pytest.mark.parametrize("entry", ["dispatch_server_dedup_kernel"])
def test_dispatch_profiler_entries_and_stable_tail(tmp_path, entry):
    # Reverse CSV order to check timestamp ordering before last-50 selection.
    rows = [f"void {entry}(),{i},{i}" for i in range(100, 0, -1)]
    rows += [f"{name},999,0" for name in (
        "dispatch_server_dedup_kernel_bf16_g8",
        "dispatch_server_dedup_kernel_unknown_g4",
        "dispatch_server_dedup_kernel_bf16_g4_suffix",
        "dispatch_server_dedup_kernel_suffix",
        "prefix_dispatch_server_dedup_kernel_bf16_g4",
        "combine_server_dedup_kernel",
        "HcomAllReduce",
    )]
    (tmp_path / "kernel_details.csv").write_text(
        "Name,Duration(us),Start Time(us)\n" + "\n".join(rows))
    samples = st.kernel_durations(tmp_path, "dispatch_server_dedup_kernel", 100)
    assert samples == list(range(1, 101))
    result = st.summarize_kernel([samples], "dispatch_server_dedup_kernel", 50)
    assert result["ranks"][0]["samples"] == 50
    assert result["ranks"][0]["mean_us"] == 75.5
    with pytest.raises(ValueError, match="expected 101"):
        st.kernel_durations(tmp_path, "dispatch_server_dedup_kernel", 101)


@pytest.mark.parametrize("second", [
    "dispatch_server_dedup_kernel [task=2]",
    "void dispatch_server_dedup_kernel()",
])
def test_dispatch_profiler_rejects_different_task_names(tmp_path, second):
    (tmp_path / "kernel_details.csv").write_text(
        "Name,Duration(us)\ndispatch_server_dedup_kernel,2\n"
        f"{second},3\n")
    with pytest.raises(ValueError, match="Mixed dispatch profiler task names"):
        st.kernel_durations(tmp_path, "dispatch_server_dedup_kernel", 2)


def test_tail_stats_and_reports(tmp_path):
    import json
    import csv

    perf = st.summarize_kernel([[100, 2, 4], [1, 4, 8]], "dispatch_server_dedup_kernel", 2)
    assert perf["slowest_rank"] == 1 and perf["max_rank_mean_kernel_us"] == 6
    assert perf["ranks"][0]["samples"] == 2 and perf["ranks"][0]["kernel_us"] == [100, 2, 4]
    result = {"case_id": 0, "config": {"B": 8}, "route": "normal", "dispatch": perf}
    st.write_reports(tmp_path, {"results": [result]})
    assert json.loads((tmp_path / "summary.json").read_text())["results"] == [result]
    with (tmp_path / "ranks.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and rows[1]["mean_us"] == "6"


def test_json_config_and_cli_override(tmp_path):
    import json

    import socket

    with socket.socket() as listener:
        listener.bind(("", 0))
        port = listener.getsockname()[1]
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "hosts": ["host0", "host1"],
                "loops": 7,
                "shmem_port": port,
                "matrix": {"num_tokens": [8, 16], "do_cpu_sync": [False, True]},
            }
        )
    )
    args = st.parse_args(["--config", str(path), "--num-tokens", "32"])
    assert args.num_iters == 7 and args.master_addr == "host0" and args.num_nodes == 2
    assert args.shmem_ip_port == f"tcp://host0:{port}"
    assert len(list(st.matrix_cases(args))) == 2 and args.num_tokens == 32
    result = subprocess.run(
        [sys.executable, st.__file__, "--config", str(path), "--plan"], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["case_count"] == 4
    path.write_text('{"matrix": {"bogus": [1]}}')
    with pytest.raises(ValueError, match="matrix supports only"):
        st.parse_args(["--config", str(path)])


def test_help_without_device_import():
    result = subprocess.run([sys.executable, st.__file__, "--help"], capture_output=True, text=True, check=True)
    assert "--skip-perf-test" in result.stdout and "--without-buffer-group" in result.stdout


def matrix_argv():
    return [
        "--num-tokens",
        "8",
        "16",
        "--dtype",
        "bf16",
        "fp8",
        "--num-experts",
        "64",
        "128",
        "--hidden",
        "4096",
        "7168",
        "--num-topk",
        "4",
        "8",
        "--do-cpu-sync",
        "0",
        "1",
    ]


def test_cartesian_product_covers_all_axes_and_scalar_cases():
    args = st.make_parser().parse_args(matrix_argv())
    cases = list(st.matrix_cases(args))
    assert len(cases) == 64
    assert len({tuple(getattr(case, key) for key in st.MATRIX_FIELDS) for case in cases}) == 64
    assert st.validate_args(args, 16) == 16
    assert all(st.validate_args(case, 16) == case.num_tokens for case in cases)
    assert cases[0].num_tokens == 8 and cases[-1].num_tokens == 16
    assert cases[0].do_cpu_sync == 0 and cases[1].do_cpu_sync == 1
    assert isinstance(args.num_tokens, list)  # Expansion must not mutate parser input.


def test_matrix_dedup_and_invalid_member():
    args = st.make_parser().parse_args(["--num-tokens", "8", "8", "16"])
    assert [case.num_tokens for case in st.matrix_cases(args)] == [8, 16]
    args = st.make_parser().parse_args(["--num-tokens", "8", "257"])
    with pytest.raises(ValueError, match="Unsupported LL combination"):
        st.validate_args(args, 16)


def test_native_preflight_dispatch_and_combine():
    calls = []
    native = SimpleNamespace(
        low_latency_layout=lambda *shape: {} if shape[2] == 4096 else None,
        low_latency_combine_tiling=lambda *shape: calls.append(shape),
    )
    args = st.make_parser().parse_args(["--dtype", "fp8", "--hidden", "4096", "7168"])
    with pytest.raises(ValueError, match="dispatch layout rejected"):
        st.validate_native_cases(native, list(st.matrix_cases(args)), 16)
    assert calls == [(16, 16, 4096, 8, 128, "bfloat16", 16)]
    native.low_latency_layout = lambda *shape: {}

    def reject_combine(*shape):
        raise ValueError("combine UB exceeded")

    native.low_latency_combine_tiling = reject_combine
    with pytest.raises(ValueError, match="combine UB exceeded"):
        st.validate_native_cases(native, list(st.matrix_cases(args)), 16)


def test_case_progress_identifies_failure(tmp_path):
    import json

    args = st.parse_args(["--num-tokens", "16", "--do-cpu-sync", "0"])
    args.case_id = 7
    args.run_directory = str(tmp_path)
    st.record_case_progress(args, 3, 16, "normal", "combine.profile")
    st.record_case_progress(args, 3, 16, "normal", "failed", RuntimeError("device failure"))
    path = tmp_path / "case_00007/normal/progress_rank_003.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["case_id"] == 7 and rows[-1]["rank"] == 3
    assert rows[-1]["failed_phase"] == "combine.profile"
    assert rows[-1]["config"]["do_cpu_sync"] is False
    assert rows[-1]["error"] == "RuntimeError: device failure"


def test_case_result_omits_raw_samples_without_changing_statistics():
    result = {"case_id": 1}
    for operation in ("dispatch", "combine"):
        result[operation] = st.summarize_kernel([[2, 4], [4, 8]], operation, 0)
    summary = st.case_result_summary(result)
    for operation in ("dispatch", "combine"):
        assert all("kernel_us" not in row for row in summary[operation]["ranks"])
        assert result[operation]["ranks"][0]["kernel_us"] == [2, 4]
        assert summary[operation]["max_rank_mean_kernel_us"] == 6
        assert summary[operation]["ranks"][0]["mean_us"] == 3


def test_explicit_endpoints_required():
    import socket

    with socket.socket() as listener:
        listener.bind(("", 0))
        port = str(listener.getsockname()[1])
    args = st.parse_args([])
    assert args.master_addr is None and args.master_port is None and args.shmem_ip_port is None
    with pytest.raises(ValueError, match="Explicit"):
        st.configure_endpoints(args, {})
    with pytest.raises(ValueError, match="Explicit"):
        st.configure_endpoints(args, {"MASTER_ADDR": "host0", "MASTER_PORT": port})
    env = {"MASTER_ADDR": "host0", "MASTER_PORT": port, "DEEPEP_SHMEM_ENDPOINT": f"tcp://host0:{port}"}
    st.configure_endpoints(args, env)
    explicit = st.parse_args(["--master-addr", "host1", "--master-port", port, "--shmem-port", port])
    st.configure_endpoints(explicit, env)
    assert env == {"MASTER_ADDR": "host1", "MASTER_PORT": port, "DEEPEP_SHMEM_ENDPOINT": f"tcp://host1:{port}"}
    explicit.master_port = 0
    with pytest.raises(ValueError, match="Invalid master_port"):
        st.configure_endpoints(explicit, {})


def test_json_fp8_scale_matrix_covers_sync_modes(tmp_path):
    import json

    config = tmp_path / "fp8.json"
    config.write_text(
        json.dumps(
            {
                "num_nodes": 2,
                "matrix": {
                    "dtype": ["fp8"],
                    "fp8_scale_dtype": ["int32", "float32"],
                    "do_cpu_sync": [0, 1],
                },
            }
        )
    )
    args = st.parse_args(["--config", str(config)])
    cases = list(st.matrix_cases(args))
    assert {(case.fp8_scale_dtype, case.do_cpu_sync) for case in cases} == {
        (scale, sync) for scale in ("int32", "float32") for sync in (0, 1)
    }
    assert all(
        st.case_config(case, st.validate_args(case, 16))["fp8_scale_dtype"] == case.fp8_scale_dtype for case in cases
    )


@pytest.mark.parametrize("iterations", [1, 40, 41, 100, 101])
def test_exp_prefix_stays_fixed_independent_of_operator_iterations(iterations):
    args = st.parse_args(["--num-iters", str(iterations)])
    assert (args.delay_exp_before, args.delay_exp_after) == (100, 0)
    st.validate_args(args, 16)


def test_delay_defaults_honor_json_and_explicit_overrides(tmp_path):
    import json

    path = tmp_path / "delay.json"
    path.write_text(json.dumps({"loops": 100, "delay_exp_before": 100}))
    args = st.parse_args(["--config", str(path)])
    assert (args.delay_exp_before, args.delay_exp_after) == (100, 0)
    st.validate_args(args, 16)
    args = st.parse_args(["--config", str(path), "--num-iters", "80", "--delay-exp-after", "7"])
    assert (args.delay_exp_before, args.delay_exp_after) == (100, 7)
    with pytest.raises(ValueError, match="Reference profiler requires"):
        st.validate_args(args, 16)


@pytest.mark.parametrize('count', [200, 0, 1, 99, 101])
def test_cannot_change_the_fixed_exp100_prefix(count):
    args = st.parse_args(['--delay-exp-before', str(count)])
    with pytest.raises(ValueError, match='fixed batch'):
        st.validate_args(args, 16)


def _wait_rows(iterations, cores, stride, table, batch_tag=1):
    rows = []
    for index in range(iterations):
        node = []
        for core in range(cores):
            slot = [0] * stride
            slot[0] = table[index] - 1_000
            slot[1] = table[index] + 10
            slot[2] = batch_tag
            slot[3] = core
            node.append(slot)
        rows.append(node)
    return rows


def test_wait_record_evidence_needs_a_128_byte_slot():
    table = [10 ** 6 * (index + 1) for index in range(4)]
    rows = _wait_rows(4, 3, 16, table)
    evidence = st.validate_wait_records(rows, iterations=4, cores=3, words=4, stride_words=16,
                                        batch_tag=1, table=table, rank=0, label="replay")
    assert len(evidence["local_release"]) == 4 and evidence["core_skew"] == [0, 0, 0, 0]

    narrow = [[slot[:4] for slot in node] for node in rows]
    with pytest.raises(RuntimeError, match="slot width"):
        st.validate_wait_records(narrow, iterations=4, cores=3, words=4, stride_words=16,
                                 batch_tag=1, table=table, rank=0, label="replay")

    missing = _wait_rows(4, 3, 16, table)
    missing[2][1][0] = -1
    with pytest.raises(RuntimeError, match="no wait record"):
        st.validate_wait_records(missing, iterations=4, cores=3, words=4, stride_words=16,
                                 batch_tag=1, table=table, rank=0, label="replay")

    stolen = _wait_rows(4, 3, 16, table)
    stolen[1][2][3] = 1
    with pytest.raises(RuntimeError, match="not this call"):
        st.validate_wait_records(stolen, iterations=4, cores=3, words=4, stride_words=16,
                                 batch_tag=1, table=table, rank=0, label="replay")

    early = _wait_rows(4, 3, 16, table)
    early[1][0][1] = table[1] - 5
    with pytest.raises(RuntimeError, match="released early"):
        st.validate_wait_records(early, iterations=4, cores=3, words=4, stride_words=16,
                                 batch_tag=1, table=table, rank=0, label="replay")

    late = _wait_rows(4, 3, 16, table)
    late[0][0][0] = table[0] + 5
    late[0][0][1] = table[0] + 10
    with pytest.raises(RuntimeError, match="entered after its deadline"):
        st.validate_wait_records(late, iterations=4, cores=3, words=4, stride_words=16,
                                 batch_tag=1, table=table, rank=0, label="replay")

    # The capture pass has no deadlines yet, so it only proves completeness.
    st.validate_wait_records(_wait_rows(4, 3, 16, table), iterations=4, cores=3, words=4,
                             stride_words=16, batch_tag=1, table=None, rank=0, label="capture")


def test_wait_table_contract_rejects_the_previous_abi():
    class Native:
        low_latency_clock_wait_record_words = 4
        low_latency_clock_wait_record_fields = ["enter", "release", "batch", "core"]
        low_latency_clock_wait_schema = 2
        low_latency_clock_wait_record_stride_words = 16
        low_latency_clock_wait_record_stride_bytes = 128
        low_latency_clock_wait_record_alignment_bytes = 128
        low_latency_clock_wait_arg_bytes = 40
        low_latency_clock_cycles_per_second = 1_000_000_000

        def low_latency_clock_wait_table(self):
            return None

    assert st.clock_wait_table_contract(Native()) == (4, 16)
    Native.low_latency_clock_wait_schema = 1
    Native.low_latency_clock_wait_record_stride_words = 4
    Native.low_latency_clock_wait_record_stride_bytes = 32
    Native.low_latency_clock_wait_record_alignment_bytes = 32
    with pytest.raises(RuntimeError, match="schema 2"):
        st.clock_wait_table_contract(Native())


def test_graph_replay_perf_requires_one_wait_before_every_dispatch(tmp_path):
    rows = []
    for index in range(3):
        rows.append(f"clock_wait_table,{index + 1},{index * 10}")
        rows.append(f"dispatch_server_dedup_kernel,{index + 2},{index * 10 + 5}")
    (tmp_path / "kernel_details.csv").write_text(
        "Name,Duration(us),Start Time(us)\n" + "\n".join(rows))
    assert st.graph_replay_perf(tmp_path, 3, "dispatch") == [2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="expected 4 each"):
        st.graph_replay_perf(tmp_path, 4, "dispatch")

    grouped = []
    for index in range(3):
        grouped.append(f"clock_wait_table,{index + 1},{index * 10}")
    for index in range(3):
        grouped.append(f"dispatch_server_dedup_kernel,{index + 2},{index * 10 + 40}")
    (tmp_path / "kernel_details.csv").write_text(
        "Name,Duration(us),Start Time(us)\n" + "\n".join(grouped))
    with pytest.raises(ValueError, match="alternate"):
        st.graph_replay_perf(tmp_path, 3, "dispatch")


def test_reference_base_is_ahead_of_every_rank():
    models = [
        {"rank": 0, "l0": 1_000, "r0": 1_000, "alpha_num": 1, "alpha_den": 1,
         "windows": [{"l": 1_000, "r": 1_000}]},
        {"rank": 1, "l0": 900, "r0": 1_200, "alpha_num": 1, "alpha_den": 1,
         "windows": [{"l": 1_100, "r": 1_400}]},
    ]
    assert st.reference_base(models, 200_000) == 1_400 + 200_000 * 1000


def test_graph_calibration_arguments_are_bounded():
    base = ["--clock-align", "--clock-align-mode", "graph", "--dispatch-only", "--do-cpu-sync", "0",
            "--num-processes", "16"]
    assert st.validate_args(st.parse_args(base), 16) == 16
    for extra in (["--clock-align-calibration-windows", "0"],
                  ["--clock-align-calibration-gap-secs", "-1"],
                  ["--clock-align-model-max-age-secs", "0"]):
        with pytest.raises(ValueError):
            st.validate_args(st.parse_args(base + extra), 16)


def _profiler_csv(path, rows):
    path.write_text("Name,Duration(us),Start Time(us)\n" + "\n".join(rows))


def test_profiler_admits_both_aot_and_jit_dispatch_names(tmp_path):
    for name in ("dispatch_server_dedup_kernel", "dispatch_jit"):
        rows = []
        for index in range(100):
            rows.append(f"clock_wait_table,1,{index * 200}")
            rows.append(f"{name},40,{index * 200 + 1}")
        _profiler_csv(tmp_path / "kernel_details.csv", rows)
        assert st.kernel_durations(tmp_path, st.DISPATCH_LOGICAL_NAME, 100) == [40.0] * 100
        assert st.graph_replay_perf(tmp_path, 100, "dispatch") == [40.0] * 100
        records = st.profiler_kernel_records(tmp_path, st.DISPATCH_PROFILER_ENTRIES)
        dispatches = [row for row in records if row["logical"] == st.DISPATCH_LOGICAL_NAME]
        assert dispatches[0]["entry"] == name and dispatches[0]["name"] == name
        assert dispatches[0]["source"].endswith("kernel_details.csv")


def test_profiler_rejects_wrong_counts_wait_gaps_and_extra_tasks(tmp_path):
    path = tmp_path / "kernel_details.csv"
    pairs = [(f"clock_wait_table,1,{i * 200}", f"dispatch_jit,40,{i * 200 + 1}") for i in range(100)]

    _profiler_csv(path, [row for pair in pairs[:99] for row in pair])
    with pytest.raises(ValueError, match="expected 100"):
        st.graph_replay_perf(tmp_path, 100, "dispatch")

    _profiler_csv(path, [pairs[i][1] for i in range(100)])
    with pytest.raises(ValueError, match="0 waits"):
        st.graph_replay_perf(tmp_path, 100, "dispatch")

    _profiler_csv(path, ["clock_wait_table,1,0", "clock_wait_table,1,10",
                         "dispatch_jit,40,20", "dispatch_jit,40,30"])
    with pytest.raises(ValueError, match="alternate"):
        st.graph_replay_perf(tmp_path, 2, "dispatch")

    doubled = [row for pair in pairs for row in pair]
    doubled += [row for pair in pairs for row in pair]
    _profiler_csv(path, doubled)
    with pytest.raises(ValueError, match="expected 100"):
        st.kernel_durations(tmp_path, st.DISPATCH_LOGICAL_NAME, 100)

    near = [f"clock_wait_table_v1,1,{i * 200}" for i in range(100)]
    near += [f"dispatch_jit_suffix,40,{i * 200 + 1}" for i in range(100)]
    _profiler_csv(path, near)
    with pytest.raises(ValueError, match="expected 100"):
        st.graph_replay_perf(tmp_path, 100, "dispatch")


def test_peer_budget_takes_the_worst_two_of_one_iteration():
    payloads = [dict(rank=0, error_ns=[4], release=[10]), dict(rank=1, error_ns=[1000], release=[20])]
    summary = st.check_submit_budget(payloads, 2, 1, None, "test", 0)
    assert summary["global_max_error_ns"] == 1000
    assert summary["pair_error_bounds_ns"] == [1004]
    assert summary["pair_error_bound_ns_max"] == 1004
    assert summary["accuracy_status"] == "reported_only"

    swapped = [dict(rank=0, error_ns=[1000], release=[10]), dict(rank=1, error_ns=[4], release=[20])]
    assert st.check_submit_budget(swapped, 2, 1, None, "test", 0)["pair_error_bounds_ns"] == [1004]

    two = [dict(rank=0, error_ns=[4, 1000], release=[1, 2]),
           dict(rank=1, error_ns=[1000, 4], release=[3, 4])]
    summary = st.check_submit_budget(two, 2, 2, None, "test", 0)
    assert summary["pair_error_bounds_ns"] == [1004, 1004]
    assert summary["pair_error_bound_ns_max"] == 1004

    single = st.check_submit_budget([dict(rank=0, error_ns=[7], release=[1])], 1, 1, None, "test", 0)
    assert single["pair_error_bounds_ns"] == [0] and single["global_max_error_ns"] == 7


def test_peer_budget_rejects_bad_payloads_and_threshold_breaches():
    good = [dict(rank=0, error_ns=[4], release=[10]), dict(rank=1, error_ns=[1000], release=[20])]
    with pytest.raises(RuntimeError, match="2 ranks"):
        st.check_submit_budget(good[:1], 2, 1, None, "test", 0)
    with pytest.raises(RuntimeError, match="rank order"):
        st.check_submit_budget([dict(rank=1), dict(rank=0)], 2, 1, None, "test", 0)
    with pytest.raises(RuntimeError, match="length"):
        st.check_submit_budget([dict(rank=0, error_ns=[4, 5]), dict(rank=1, error_ns=[1])],
                               2, 1, None, "test", 0)
    with pytest.raises(RuntimeError, match="invalid"):
        st.check_submit_budget([dict(rank=0, error_ns=[-4]), dict(rank=1, error_ns=[1])],
                               2, 1, None, "test", 0)
    with pytest.raises(RuntimeError, match="exceeds"):
        st.check_submit_budget(good, 2, 1, 1000, "test", 0, replay_id=2)


def _fake_windows(rank, count=3):
    base = 10 ** 12
    span = 5 * 10 ** 9
    return [dict(window=index, rank=rank, l=base + index * span, r=base + index * span,
                 error_cycles=9_000, round_trip_cycles=18_000, sequence=0,
                 valid_samples=64, rejected_samples=0) for index in range(count)]


def test_model_admission_names_the_failing_phase():
    models = [st.ll_clock_model.fit_model(_fake_windows(rank), rank) for rank in range(2)]
    st.validate_clock_models(models, 2, "reanchor", 1, 2)
    models[1] = dict(rank=1)
    with pytest.raises(RuntimeError, match=r"phase=reanchor rank=1 replay=2.*missing"):
        st.validate_clock_models(models, 2, "reanchor", 1, 2)


def test_capture_records_are_only_written_by_the_probe_replay():
    table = [10 ** 6 * (index + 1) for index in range(2)]
    records = [[[-1] * 16 for _ in range(2)] for _ in range(2)]

    def clear():
        for node in records:
            for slot in node:
                slot[:] = [-1] * 16

    def replay():
        for index, node in enumerate(records):
            for core, slot in enumerate(node):
                slot[0] = table[index] - 1000
                slot[1] = table[index] + 10
                slot[2] = 1
                slot[3] = core

    st.assert_records_untouched(records, 0, "capture")
    probe = st.probe_capture_records(clear, replay, lambda: records, iterations=2, cores=2,
                                     words=4, stride_words=16, batch_tag=1, rank=0)
    assert len(probe["local_release"]) == 2

    def missing():
        replay()
        records[1][1][0] = -1

    with pytest.raises(RuntimeError, match="probe_replay"):
        st.probe_capture_records(clear, missing, lambda: records, iterations=2, cores=2,
                                 words=4, stride_words=16, batch_tag=1, rank=0)

    written = [[[-1] * 16 for _ in range(2)] for _ in range(2)]
    written[0][0][0] = 5
    with pytest.raises(RuntimeError, match="already has a record"):
        st.assert_records_untouched(written, 0, "capture")


def test_graph_output_checks_every_node_against_the_route():
    entries = [[(0, 0), (0, 1), (1, 0), (1, 1)], [(2, 0)]]
    # Two local experts times three source ranks: statuses are expert-major and
    # cumulative, which is neither the per-expert count nor a padded row count.
    expected = [2, 4, 4, 4, 4, 5]
    assert st.expected_send_counts(entries, 3) == expected

    class Counts:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return SimpleNamespace(tolist=lambda: list(self.values))

    class Handle:
        def __init__(self, values):
            self.send_counts = Counts(values)

    captured = [(None, None, None, Handle(expected), None) for _ in range(3)]
    st.verify_graph_outputs(captured, expected, None, 0, 0)

    stale = [(None, None, None, Handle(expected), None) for _ in range(2)]
    stale[1] = (None, None, None, Handle([-1] * len(expected)), None)
    with pytest.raises(AssertionError, match="replay 1 node 1"):
        st.verify_graph_outputs(stale, expected, None, 0, 1)

    mixed = [(None, None, None, Handle(expected), None) for _ in range(3)]
    mixed[1] = (None, None, None, Handle([2, 4, 4, 4, 4, 4]), None)
    with pytest.raises(AssertionError, match="node 1 count output differs at 5"):
        st.verify_graph_outputs(mixed, expected, None, 0, 0)


def test_profiler_replays_inside_the_active_step(tmp_path, monkeypatch):
    events = []
    state = {"action": "none"}

    class FakeProfile:
        def __enter__(self):
            state["action"] = "active"
            events.append("start_trace")
            return self

        def __exit__(self, *exc):
            events.append("finalize")
            state["action"] = "none"
            return False

        def step(self):
            events.append(f"step:{state['action']}")
            if state["action"] == "active":
                state["action"] = "none"

    class FakeGraph:
        def replay(self):
            events.append(f"replay:{state['action']}")

    class FakeStreamContext:
        def __enter__(self):
            events.append("stream_enter")
            return None

        def __exit__(self, *exc):
            return False

    fake_torch = SimpleNamespace(npu=SimpleNamespace(
        stream=lambda stream: FakeStreamContext(),
        synchronize=lambda: events.append("synchronize")))
    fake_npu = SimpleNamespace(profiler=SimpleNamespace(
        profile=lambda **kwargs: FakeProfile(),
        schedule=lambda **kwargs: None,
        tensorboard_trace_handler=lambda path: None,
        _ExperimentalConfig=lambda **kwargs: None,
        AiCMetrics=SimpleNamespace(PipeUtilization=None),
        ProfilerLevel=SimpleNamespace(Level1=None),
        ExportType=SimpleNamespace(Text=None)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    monkeypatch.setattr(st, "graph_replay_perf", lambda *args: ["extracted"])

    assert st.replay_with_profile(FakeGraph(), object(), tmp_path / "profile", 100, "dispatch") == ["extracted"]
    assert events == ["start_trace", "stream_enter", "replay:active", "synchronize",
                      "step:active", "finalize"]

    events.clear()
    state["action"] = "none"
    with FakeProfile():
        FakeProfile().step()
        FakeGraph().replay()
    assert "replay:none" in events
