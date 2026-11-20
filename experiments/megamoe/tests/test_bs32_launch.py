# Copyright (c) 2026, Lu Lu

"""Check persisted settings and build/run separation without CANN or devices."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("bs32_launcher", ROOT / "experiments/megamoe/scripts/bs32.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def saved(tmp_path):
    cann = tmp_path / "CANN with spaces" / "set_env.sh"
    cann.parent.mkdir()
    cann.write_text("export ASCEND_HOME_PATH=/opt/cann\n")
    config = dict(
        launcher.defaults(),
        node_rank=0,
        master_addr="192.0.2.10",
        cann_env=str(cann),
        rdma_backend="HNS_1825",
        case_dir=str(tmp_path / "results"),
    )
    path = tmp_path / "node.json"
    launcher.save_config(path, config)
    return path, config


def test_first_configuration_and_explicit_partial_update(tmp_path, saved):
    original_path, original = saved
    path = tmp_path / "other-host.json"
    launcher.main(
        [
            "configure",
            "--config",
            str(path),
            "--node-rank",
            "1",
            "--master-addr",
            original["master_addr"],
            "--cann-env",
            original["cann_env"],
            "--rdma-backend",
            "HNS_1825",
        ]
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["node_rank"] == 1
    assert json.loads(original_path.read_text(encoding="utf-8")) == original
    launcher.main(["configure", "--config", str(path), "--master-addr", "192.0.2.20"])
    assert json.loads(path.read_text(encoding="utf-8")) == dict(config, master_addr="192.0.2.20")


def test_repeated_run_reuses_settings_without_prompt_or_environment_override(saved, monkeypatch):
    path, config = saved
    before = path.read_bytes()
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("Prompt on repeat run"))
    monkeypatch.setenv("MASTER_ADDR", "192.0.2.99")
    monkeypatch.setenv("NODE_RANK", "1")
    calls = []
    monkeypatch.setattr(launcher, "dispatch", lambda *a, **kw: calls.append((a, kw)))
    for _ in range(2):
        launcher.main(["run", "--config", str(path)])
    assert calls == [(("run", config), {"rebuild_shmem": False, "errors_only": False})] * 2
    assert path.read_bytes() == before


def test_interactive_config_keeps_existing_values_on_enter(saved, monkeypatch):
    path, config = saved
    monkeypatch.setattr(launcher.sys.stdin, "isatty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda label: prompts.append(label) or "")
    launcher.main(["configure", "--config", str(path)])
    assert len(prompts) == 4
    assert json.loads(path.read_text(encoding="utf-8")) == config


@pytest.mark.parametrize("failure", ["preflight", "golden"])
def test_failed_preparation_never_launches_benchmark(saved, monkeypatch, failure):
    _, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    calls = []

    def preflight(*args):
        if failure == "preflight":
            raise RuntimeError("missing installation")

    def execute(command, *args, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(launcher, "preflight", preflight)
    monkeypatch.setattr(launcher, "execute", execute)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        launcher.dispatch("run", config)
    assert len(calls) == (1 if failure == "golden" else 0)
    assert all("torch.distributed.run" not in command for command in calls)


def test_isolated_cli_help():
    result = subprocess.run([sys.executable, "-I", launcher.__file__, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "configure,show-config,build,golden,run" in result.stdout
    assert "perf,trace" in result.stdout
    assert "--timer-ranks" in result.stdout


@pytest.mark.parametrize(
    "options",
    [
        ["--node-rank", "2"],
        ["--master-addr", "127.0.0.1"],
        ["--master-port", "26733"],
        ["--golden-workers", "0"],
        ["--rdma-backend", "NONE"],
        ["--cann-env", "/missing/set_env.sh"],
    ],
)
def test_bad_edit_preserves_previous_configuration(saved, options):
    path, _ = saved
    before = path.read_bytes()
    with pytest.raises(ValueError):
        launcher.main(["configure", "--config", str(path), *options])
    assert path.read_bytes() == before


def test_runtime_cannot_mutate_config_or_rebuild_sdk(saved):
    path, _ = saved
    for options in (["--master-addr", "192.0.2.20"], ["--rebuild-shmem"], ["--errors-only"]):
        with pytest.raises(SystemExit):
            launcher.main(["run", "--config", str(path), *options])


@pytest.mark.parametrize("action,expected", [("build", 1), ("golden", 1), ("run", 2)])
def test_stage_separation_and_fixed_case(saved, monkeypatch, action, expected):
    _, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {"TEST": "yes"})
    probes = []
    monkeypatch.setattr(launcher, "preflight", lambda *args: probes.append(args))
    calls = []
    monkeypatch.setattr(launcher, "execute", lambda *args, **kwargs: calls.append((args, kwargs)))
    launcher.dispatch(action, config, rebuild_shmem=action == "build")
    assert len(calls) == expected
    if action == "build":
        assert calls[0][0][0] == [
            "bash",
            str(ROOT / "experiments/megamoe/scripts/build.sh"),
            "--rdma-backend",
            "HNS_1825",
            "--rebuild-shmem",
        ]
        assert not probes
        log = Path(calls[0][0][2])
        assert log.parent.parent == Path(config["case_dir"])
        assert log.name == "build.log"
        assert calls[0][0][1]["PIP_VERBOSE"] == "1"
        assert calls[0][1] == {"cwd": ROOT, "errors_only": False}
    else:
        assert len(probes) == 1
        golden = calls[0][0][0]
        assert golden[:4] == [config["python"], "-I", "-m", "deep_ep_experimental.megamoe.golden"]
        assert golden[golden.index("--ranks") + 1] == "0-15"
        case_path = Path(golden[golden.index("--config") + 1])
        assert json.loads(case_path.read_text(encoding="utf-8")) == launcher.CASE
        assert launcher.CASE["bs_values"] == [32]
        assert launcher.CASE["model"] == dict(
            hidden=3584, intermediate_hidden=1280, num_experts=384, num_topk=8, num_shared_experts=0
        )
        if action == "run":
            command = calls[1][0][0]
            assert command[:4] == [config["python"], "-I", "-m", "torch.distributed.run"]
            assert command[command.index("--nproc-per-node") + 1] == "16"
            assert command[command.index("--golden-dir") + 1] == golden[golden.index("--golden-dir") + 1]
            assert command[command.index("--config") + 1] == str(case_path)


def test_second_host_rank_range_and_saved_python(saved):
    _, config = saved
    config = dict(config, node_rank=1, python="/opt/env with spaces/bin/python")
    golden, benchmark = launcher.test_commands(config, Path(config["case_dir"]))
    assert golden[golden.index("--ranks") + 1] == "16-31"
    assert benchmark[benchmark.index("--node-rank") + 1] == "1"
    assert golden[0] == benchmark[0] == config["python"]


def test_environment_uses_positional_paths_and_removes_import_overrides(saved, monkeypatch):
    _, config = saved
    for name in ("SHMEM_ROOT", "CATLASS_ROOT", "PYTHONPATH", "PYTHONHOME", "PYTHON_EXEC"):
        monkeypatch.setenv(name, "unwanted")
    before = dict(os.environ)

    def source(command, *, env, cwd, check, text, stdout):
        assert command[0:2] == ["bash", "-c"]
        assert command[4:6] == [config["cann_env"], config["python"]]
        assert config["cann_env"] not in command[2]
        assert "PYTHONPATH" not in env and "SHMEM_ROOT" not in env
        return subprocess.CompletedProcess(command, 0, json.dumps(dict(env, ASCEND_HOME_PATH="/opt/cann")))

    monkeypatch.setattr(launcher.subprocess, "run", source)
    env = launcher.environment(config)
    assert env["MASTER_ADDR"] == config["master_addr"]
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, range(16)))
    assert env["DEEPEP_SHMEM_ENDPOINT"] == "tcp://192.0.2.10:26733"
    assert env["DEEPEP_MEGAMOE_TIMER"] == "OFF"
    assert dict(os.environ) == before


def test_logged_child_failure_is_not_reported_as_success(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(subprocess.CalledProcessError) as error:
        launcher.execute(
            [sys.executable, "-I", "-c", "print('test failure'); raise SystemExit(7)"],
            dict(os.environ),
            log,
            cwd=tmp_path,
        )
    assert error.value.returncode == 7
    assert "test failure" in log.read_text(encoding="utf-8")


def test_errors_only_preserves_diagnostics_full_log_and_exit_code(tmp_path, capsys):
    # Model noisy Clang/CANN diagnostics, including template notes and a warning promoted to error.
    lines = [f"warning flood {index}: warning: deprecated API" for index in range(50)]
    lines += [
        "In file included from kernel.cpp:13:",
        "\x1b[31mkernel.hpp:724:9: error: no matching function for call to 'CombineWaveTokenRange'\x1b[0m",
        "  CombineWaveTokenRange(output);",
        "  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~",
        "stage.hpp:459:24: note: candidate not viable: would lose const qualifier",
        "kernel.hpp:730:9: error: unused variable [-Werror,-Wunused-variable]",
        "gmake: *** [Makefile:136: all] Error 2",
    ]
    log = tmp_path / "build.log"
    code = "\n".join([f"print({line!r})" for line in lines] + ["raise SystemExit(2)"])
    # Avoid printing the test's embedded warning strings as part of the [EXEC] command.
    child = tmp_path / "compiler.py"
    child.write_text(code, encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError) as error:
        launcher.execute([sys.executable, "-I", str(child)], dict(os.environ), log, cwd=tmp_path, errors_only=True)
    assert error.value.returncode == 2
    output = capsys.readouterr().out
    assert "warning flood 0:" not in output
    assert "In file included from kernel.cpp:13:" in output
    assert "error: no matching function" in output
    assert "would lose const qualifier" in output
    assert "[-Werror,-Wunused-variable]" in output
    assert "Error 2" in output
    assert log.read_text(encoding="utf-8").splitlines() == lines


def test_errors_only_success_and_unknown_failure_keep_full_logs(saved, tmp_path, monkeypatch, capsys):
    log = tmp_path / "warnings.log"
    child = tmp_path / "warnings.py"
    child.write_text("print('file.cpp:1: warning: deprecated API')\n", encoding="utf-8")
    launcher.execute([sys.executable, "-I", str(child)], dict(os.environ), log, cwd=tmp_path, errors_only=True)
    assert "warning: deprecated" not in capsys.readouterr().out
    assert "warning: deprecated" in log.read_text(encoding="utf-8")

    _, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})

    def fail(command, env, log, **kwargs):
        assert kwargs["errors_only"]
        Path(log).write_text("custom failure without a recognized diagnostic\n", encoding="utf-8")
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(launcher, "execute", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        launcher.dispatch("build", config, errors_only=True)
    assert error.value.returncode == 7
    assert "full log:" in capsys.readouterr().err


@pytest.mark.parametrize("node,action", [(0, "perf"), (1, "perf"), (0, "trace"), (1, "trace")])
@pytest.mark.parametrize("method", ["legacy", "shmem"])
def test_sampling_workflow_reuses_settings_and_skips_golden(saved, monkeypatch, node, action, method):
    path, config = saved
    config["node_rank"] = node
    launcher.save_config(path, config)
    before = path.read_bytes()
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    monkeypatch.setattr(launcher, "preflight", lambda *args: None)
    calls = []
    outputs = []

    def execute(command, env, log, **kwargs):
        calls.append(command)
        assert command[0] == config["python"]
        assert env["DEEPEP_MEGAMOE_TIMER"] == ("ON" if action == "trace" else "OFF")
        if "torch.distributed.run" in command:
            assert command[command.index("--node-rank") + 1] == str(node)
            assert command[command.index("--mode") + 1] == action
            assert not any(arg.startswith(("--combine-", "--dispatch-", "--reduce-")) for arg in command)
            assert command[command.index("--perf-method") + 1] == method
            assert command[command.index("--input") + 1] == "fp8"
            assert command[command.index("--routing") + 1] == "balanced"
            assert command[command.index("--nproc-per-node") + 1] == "16"
            case = json.loads(Path(command[command.index("--config") + 1]).read_text())
            warmup, iters = (2, 1) if action == "trace" else (20, 100)
            assert case == dict(launcher.CASE, warmup=warmup, iters=iters)
            output = Path(command[command.index("--output-dir") + 1])
            outputs.append(output)
            if action == "trace":
                assert "--timer" in command
                assert command[command.index("--timer-ranks") + 1] == "0,16"
            else:
                assert "--timer" not in command
                report_dir = output / f"rank_{node * 16}" / "bs_32"
                report_dir.mkdir(parents=True)
                (report_dir / "profiling.json").write_text(
                    json.dumps(
                        dict(
                            implementation="dispatch_qp8_combine_qp16",
                            rounds=50,
                            recorded_rounds=100,
                            mean_us=12.5,
                            rank_max_mean_us=25.0,
                        )
                    )
                )
        elif str(launcher.ROOT / "experiments/megamoe/scripts/trace_text.py") in command:
            assert command[-3:] == ["--max-bytes", "65535", "--print-parts"]
            assert command[command.index("export") + 1] == str(outputs[-1])
        else:
            assert "deep_ep_experimental.megamoe.timer" in command
            assert command[command.index("--ranks") + 1] == str(node * 16)
            assert "--include-wait" not in command and "--merge" in command

    monkeypatch.setattr(launcher, "execute", execute)
    for _ in range(2):
        launcher.main([action, "--config", str(path), "--perf-method", method])
    assert json.loads((outputs[-1] / "run.json").read_text())["perf_method"] == method
    assert json.loads((outputs[-1] / "run.json").read_text())["implementation"] == "dispatch_qp8_combine_qp16"
    assert len(calls) == (6 if action == "trace" else 2)
    assert outputs[0] != outputs[1]
    summary = (outputs[-1] / "summary.txt").read_text()
    if action == "perf":
        assert "minimum rank: 12.500 us" in summary and "maximum rank: 25.000 us" in summary
        assert "50/100 rounds" in summary
        assert "not a cross-device timestamp span" in summary
    else:
        assert "[TRACE SUMMARY]" in summary
        assert "Warmup calls: 2; trace batch calls: 1" in summary
        assert "Performance profiler disabled" in summary
        assert f"Local traced ranks: [{node * 16}]" in summary
        assert not list(outputs[-1].rglob("profiling.json"))
    assert path.read_bytes() == before
    assert launcher.CASE["warmup"] == 0 and launcher.CASE["iters"] == 1


@pytest.mark.parametrize("flag,include", [(None, False), ("--hide-wait", False), ("--include-wait", True)])
def test_trace_wait_option_reaches_parser_and_run_manifest(saved, monkeypatch, flag, include):
    path, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    monkeypatch.setattr(launcher, "preflight", lambda *args: None)
    monkeypatch.setattr(launcher, "performance_summary", lambda *args: pytest.fail("trace entered performance summary"))
    calls = []
    monkeypatch.setattr(launcher, "execute", lambda command, *args, **kwargs: calls.append(command))
    launcher.main(["trace", "--config", str(path), *([flag] if flag else [])])
    assert len(calls) == 3
    assert calls[2][-3:] == ["--max-bytes", "65535", "--print-parts"]
    assert ("--include-wait" in calls[1]) is include
    run_dir = Path(calls[0][calls[0].index("--output-dir") + 1])
    assert json.loads((run_dir / "run.json").read_text())["include_wait"] is include


def test_timer_build_is_explicit_and_does_not_launch_workers(saved, monkeypatch):
    path, _ = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    calls = []
    monkeypatch.setattr(launcher, "execute", lambda *args, **kwargs: calls.append((args, kwargs)))
    launcher.main(["build", "--timer", "--errors-only", "--config", str(path)])
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0][0] == "bash" and args[1]["DEEPEP_MEGAMOE_TIMER"] == "ON"
    assert kwargs["errors_only"]
    launcher.main(["build", "--config", str(path)])
    assert calls[-1][0][1]["DEEPEP_MEGAMOE_TIMER"] == "OFF"


def test_trace_round_overrides_and_host_without_selected_ranks(saved, monkeypatch):
    path, _ = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    monkeypatch.setattr(launcher, "preflight", lambda *args: None)
    calls = []
    monkeypatch.setattr(launcher, "execute", lambda *args, **kwargs: calls.append(args))
    summaries = []
    monkeypatch.setattr(launcher, "trace_summary", lambda *args: summaries.append(args))
    launcher.main(["trace", "--config", str(path), "--warmup", "0", "--iters", "5", "--timer-ranks", "16-18,31"])
    assert len(calls) == 1  # node0 still joins the benchmark; no local timer files to parse.
    command = calls[0][0]
    case = json.loads(Path(command[command.index("--config") + 1]).read_text())
    assert (case["warmup"], case["iters"]) == (0, 5)
    assert command[command.index("--timer-ranks") + 1] == "16,17,18,31"
    assert summaries[0][-1] == []


@pytest.mark.parametrize("failure", ["preflight", "benchmark", "parse"])
def test_failed_trace_does_not_print_done_or_summary(saved, monkeypatch, capsys, failure):
    _, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})

    def preflight(*args):
        if failure == "preflight":
            raise RuntimeError("build --timer first")

    calls = []

    def execute(command, *args, **kwargs):
        calls.append(command)
        if failure == "benchmark" or "deep_ep_experimental.megamoe.timer" in command:
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(launcher, "preflight", preflight)
    monkeypatch.setattr(launcher, "execute", execute)
    monkeypatch.setattr(launcher, "trace_summary", lambda *args: pytest.fail("summary after failed trace"))
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        launcher.dispatch("trace", config)
    assert len(calls) == {"preflight": 0, "benchmark": 1, "parse": 2}[failure]
    assert "[DONE]" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--extra-bs"],
        ["trace", "--extra-bs"],
        ["build", "--extra-bs"],
        ["perf", "--timer"],
        ["perf", "--include-wait"],
        ["build", "--hide-wait"],
        ["trace", "--include-wait", "--hide-wait"],
        ["perf", "--timer-ranks", "0"],
        ["run", "--warmup", "20"],
        ["build", "--iters", "100"],
        ["trace", "--warmup", "-1"],
        ["perf", "--iters", "0"],
        ["run", "--perf-method", "shmem"],
        ["perf", "--perf-method", "unknown"],
        ["trace", "--timer-ranks", "32"],
        ["trace", "--timer-ranks", "1-0"],
        ["trace", "--timer-ranks", "0,0"],
        ["trace", "--timer-ranks", "0-999999999"],
    ],
)
def test_invalid_performance_options_fail_before_launch(saved, monkeypatch, arguments):
    path, _ = saved
    monkeypatch.setattr(launcher, "dispatch", lambda *args, **kwargs: pytest.fail("invalid arguments launched"))
    with pytest.raises(SystemExit):
        launcher.main([*arguments, "--config", str(path)])


@pytest.mark.parametrize("max_bs,limit", [(32, 32), (256, 32), (256, 144), (256, 256)])
@pytest.mark.parametrize("requested,compiled", [(False, False), (False, True), (True, False), (True, True)])
def test_preflight_checks_installed_timer_support(saved, tmp_path, monkeypatch, requested, compiled, max_bs, limit):
    _, config = saved
    checkout = tmp_path / "checkout"
    library = checkout / "third-party/shmem/install/shmem/lib/libshmem.so"
    library.parent.mkdir(parents=True)
    library.touch()
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "_build_info.json").write_text(
        json.dumps(dict(megamoe=dict(enabled=True), shmem=dict(rdma_backend=config["rdma_backend"])))
    )
    monkeypatch.setattr(launcher, "ROOT", checkout)
    monkeypatch.setattr("importlib.resources.files", lambda package: installed)
    modules = {}
    for name in (
        "torch",
        "torch_npu",
        "deep_ep",
        "deep_ep._native",
        "deep_ep_experimental",
        "deep_ep_experimental.megamoe",
        "deep_ep_experimental.megamoe._ep32_C",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["torch"].npu = SimpleNamespace(device_count=lambda: 16)
    modules["deep_ep"].__file__ = str(installed / "__init__.py")
    modules["deep_ep._native"].require_native = lambda operation: None
    modules["deep_ep_experimental.megamoe._ep32_C"].timer_enabled = compiled
    modules["deep_ep_experimental.megamoe._ep32_C"].hierarchical_max_tokens = limit
    modules["deep_ep_experimental.megamoe._ep32_C"].api_protocol = 17

    def execute(command, env):
        monkeypatch.setattr(sys, "argv", ["-c", *command[4:]])
        exec(command[3], {})

    monkeypatch.setattr(launcher, "execute", execute)
    env = dict(DEEPEP_MEGAMOE_TIMER="ON" if requested else "OFF")
    if requested and not compiled:
        with pytest.raises(RuntimeError, match="build --timer"):
            launcher.preflight(config, env, max_bs=max_bs)
    elif max_bs > limit:
        with pytest.raises(RuntimeError, match="run run_bs32.sh build"):
            launcher.preflight(config, env, max_bs=max_bs)
    else:
        launcher.preflight(config, env, max_bs=max_bs)


@pytest.mark.parametrize("node", [0, 1])
@pytest.mark.parametrize("method", ["legacy", "shmem"])
def test_extra_batches_keep_independent_capacity_and_aggregate_reports(saved, monkeypatch, node, method):
    path, config = saved
    config["node_rank"] = node
    launcher.save_config(path, config)
    before = path.read_bytes()
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    probes, outputs, batches = [], [], []
    monkeypatch.setattr(launcher, "preflight", lambda *a, **kw: probes.append(kw))

    def execute(command, env, log):
        assert "torch.distributed.run" in command
        assert "--timer" not in command and env["DEEPEP_MEGAMOE_TIMER"] == "OFF"
        for option, value in (
            ("--mode", "perf"),
            ("--perf-method", method),
            ("--node-rank", str(node)),
            ("--input", "fp8"),
            ("--routing", "balanced"),
        ):
            assert command[command.index(option) + 1] == value
        case_path = Path(command[command.index("--config") + 1])
        case = json.loads(case_path.read_text())
        assert len(case["bs_values"]) == 1  # benchmark allocates max(bs_values)
        bs = case["bs_values"][0]
        assert case == dict(launcher.CASE, bs_values=[bs], warmup=3, iters=8)
        output = Path(command[command.index("--output-dir") + 1])
        assert output == case_path.parent and output.name == f"bs_{bs}"
        assert log == output / "console.log"
        outputs.append(output)
        batches.append(bs)
        report_dir = output / f"rank_{node * 16}" / f"bs_{bs}"
        report_dir.mkdir(parents=True)
        (report_dir / "profiling.json").write_text(
            json.dumps(
                dict(
                    implementation="dispatch_qp8_combine_qp16",
                    perf_method=method,
                    rounds=4,
                    recorded_rounds=8,
                    mean_us=bs * 1.5,
                    rank_max_mean_us=bs * 2.5,
                )
            )
        )

    monkeypatch.setattr(launcher, "execute", execute)
    returned = launcher.main(
        ["perf", "--extra-bs", "--config", str(path), "--warmup", "3", "--iters", "8", "--perf-method", method]
    )
    assert probes == [{"max_bs": 256}]
    assert batches == [32, 1, 4, 8, 16, 256]
    assert len(set(outputs)) == 6 and len({p.parent for p in outputs}) == 1
    root = outputs[0].parent
    assert returned == root
    assert json.loads((root / "case.json").read_text())["bs_values"] == batches
    run = json.loads((root / "run.json").read_text())
    assert run["extra_bs"] and run["bs_values"] == batches
    rows = list(csv.DictReader((root / "summary.csv").read_text().splitlines()))
    assert [int(row["bs"]) for row in rows] == batches
    for row, bs in zip(rows, batches):
        assert float(row["rank_max_mean_us"]) == bs * 2.5
        assert row["perf_method"] == method and row["rounds"] == "4"
        assert f"EP32 BS{bs};" in (root / "summary.txt").read_text()
    assert path.read_bytes() == before and launcher.CASE["bs_values"] == [32]


def test_extra_batches_stop_on_failure_without_success_summary(saved, monkeypatch, capsys):
    _, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    monkeypatch.setattr(launcher, "preflight", lambda *a, **kw: None)
    batches, outputs = [], []

    def execute(command, env, log):
        case = json.loads(Path(command[command.index("--config") + 1]).read_text())
        batches.extend(case["bs_values"])
        outputs.append(Path(command[command.index("--output-dir") + 1]))
        if batches[-1] == 4:
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(launcher, "execute", execute)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.dispatch("perf", config, extra_bs=True)
    assert batches == [32, 1, 4]
    assert not (outputs[0].parent / "summary.txt").exists()
    assert "[DONE]" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "option",
    [
        "--combine-reduce-qps",
        "--dispatch-workers",
        "--reduce-lane-init",
        "--combine-sparse-scan",
        "--reduce-double-buffer",
        "--combine-single-row",
        "--combine-ready-trace",
        "--dispatch-skip-empty",
        "--dispatch-lookahead",
        "--dispatch-single-row",
        "--combine-cross-expert",
    ],
)
def test_perf_rejects_removed_algorithm_options(saved, monkeypatch, option):
    path, _ = saved
    monkeypatch.setattr(launcher, "dispatch", lambda *a, **kw: pytest.fail("Retired option launched"))
    with pytest.raises(SystemExit):
        launcher.main(["perf", "--config", str(path), option])


@pytest.mark.parametrize("action", ["run", "golden", "perf", "trace"])
@pytest.mark.parametrize("bs", [72, 128])
def test_explicit_batch_reaches_preflight_case_and_trace_paths(saved, monkeypatch, action, bs):
    path, config = saved
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "environment", lambda _: {})
    probes, calls, summaries = [], [], []
    monkeypatch.setattr(launcher, "preflight", lambda *args, **kwargs: probes.append(kwargs))
    monkeypatch.setattr(launcher, "execute", lambda *args, **kwargs: calls.append(args[0]))
    monkeypatch.setattr(launcher, "performance_summary", lambda *args: summaries.append(args))
    run_dir = launcher.main([action, "--config", str(path), "--bs", str(bs)], print_text=False)
    assert probes == [{"max_bs": bs}]
    case = json.loads((run_dir / "case.json").read_text(encoding="utf-8"))
    assert case["bs_values"] == [bs]
    assert launcher.CASE["bs_values"] == [32]
    if action == "trace":
        parser = next(command for command in calls if "deep_ep_experimental.megamoe.timer" in command)
        assert str(run_dir / f"bs_{bs}" / "timer") in parser
        summary = (run_dir / "summary.txt").read_text(encoding="utf-8")
        assert f"BS{bs}" in summary and str(run_dir / f"bs_{bs}" / "timer") in summary
    if action == "perf":
        assert summaries[0][-1] == [bs]


@pytest.mark.parametrize(
    "args",
    [["perf", "--bs", "0"], ["perf", "--bs", "257"], ["build", "--bs", "72"], ["perf", "--bs", "72", "--extra-bs"]],
)
def test_invalid_explicit_batch_rejected_before_launch(args):
    with pytest.raises(SystemExit):
        launcher.main(args)
