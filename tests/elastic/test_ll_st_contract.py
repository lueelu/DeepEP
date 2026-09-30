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


@pytest.mark.parametrize("iterations,expected", [(1, 20), (40, 20), (41, 20), (100, 50), (101, 50)])
def test_delay_defaults_follow_operator_iterations(iterations, expected):
    args = st.parse_args(["--num-iters", str(iterations)])
    assert args.delay_exp_before == args.delay_exp_after == expected


def test_delay_defaults_honor_json_and_explicit_overrides(tmp_path):
    import json

    path = tmp_path / "delay.json"
    path.write_text(json.dumps({"loops": 100, "delay_exp_before": 0}))
    args = st.parse_args(["--config", str(path)])
    assert (args.delay_exp_before, args.delay_exp_after) == (0, 50)
    args = st.parse_args(["--config", str(path), "--num-iters", "80", "--delay-exp-after", "7"])
    assert (args.delay_exp_before, args.delay_exp_after) == (0, 7)
