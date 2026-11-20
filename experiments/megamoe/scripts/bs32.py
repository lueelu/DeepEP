# Copyright (c) 2026, Lu Lu

"""Persist per-host settings and build, check, profile or trace the two-host EP32 case (BS32 by default)."""

import argparse
from collections import deque
import csv
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / ".megamoe-bs32" / f"{socket.gethostname()}.json"
CASE = {
    "schema_version": 1,
    "model": {"hidden": 3584, "intermediate_hidden": 1280, "num_experts": 384, "num_topk": 8, "num_shared_experts": 0},
    "bs_values": [32],
    "warmup": 0,
    "iters": 1,
    "seed": 20260922,
}
EXTRA_PERF_BATCHES = (1, 4, 8, 16, 256)

FIELDS = (
    "node_rank",
    "master_addr",
    "cann_env",
    "rdma_backend",
    "python",
    "case_dir",
    "master_port",
    "shmem_port",
    "golden_workers",
)


def defaults():
    return dict(
        schema_version=1,
        node_rank=None,
        master_addr="",
        cann_env="",
        rdma_backend="",
        python=sys.executable,
        case_dir=str(ROOT / "benchmark-runs/bs32"),
        master_port=29654,
        shmem_port=26733,
        golden_workers=4,
    )


def validate(config):
    if set(config) != {"schema_version", *FIELDS} or config["schema_version"] != 1:
        raise ValueError("Unsupported BS32 config fields/schema; use configure")
    if type(config["node_rank"]) is not int or config["node_rank"] not in (0, 1):
        raise ValueError("node_rank must be 0 or 1")
    address = ipaddress.IPv4Address(config["master_addr"])
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise ValueError("master_addr must be node 0's IPv4 address reachable from both hosts")
    for name in ("master_port", "shmem_port"):
        if type(config[name]) is not int or not 1024 <= config[name] <= 65535:
            raise ValueError(f"{name} must be an integer in [1024, 65535]")
    if config["master_port"] == config["shmem_port"]:
        raise ValueError("master_port and shmem_port must differ")
    if type(config["golden_workers"]) is not int or config["golden_workers"] < 1:
        raise ValueError("golden_workers must be positive")
    if config["rdma_backend"] != "HNS_1825":
        raise ValueError("rdma_backend must be HNS_1825")
    for name in ("cann_env", "python", "case_dir"):
        if not isinstance(config[name], str) or not Path(config[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    for name in ("cann_env", "python"):
        if not Path(config[name]).is_file():
            raise ValueError(f"{name} does not exist: {config[name]}")


def save_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configure(path, args):
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else defaults()
    overrides = {name: getattr(args, name) for name in FIELDS if getattr(args, name) is not None}
    config.update(overrides)
    if not overrides:
        if not sys.stdin.isatty():
            raise ValueError(
                "No interactive terminal; use configure --node-rank ... --master-addr ... --cann-env ... --rdma-backend ..."
            )
        for name, label in (
            ("node_rank", "本机节点编号（0 或 1）"),
            ("master_addr", "第一台机器的 IP"),
            ("cann_env", "CANN set_env.sh 完整路径"),
            ("rdma_backend", "RoCE 后端（HNS_1825）"),
        ):
            value = input(f"{label} [{config[name] if config[name] is not None else ''}]: ").strip()
            if value:
                config[name] = int(value) if name == "node_rank" else value
    # Keep the venv/Conda executable path, even when it is a symlink.
    executable = shutil.which(os.path.expanduser(str(config["python"])))
    if executable is None:
        raise ValueError(f"Python executable not found: {config['python']}")
    config["python"] = os.path.abspath(executable)
    for name in ("cann_env", "case_dir"):
        if config[name]:
            config[name] = str(Path(config[name]).expanduser().absolute())
    validate(config)
    save_config(path, config)
    print(f"[CONFIG] Saved: {path}\n{json.dumps(config, indent=2, ensure_ascii=False)}", flush=True)
    return config


def environment(config):
    env = dict(os.environ)
    # Use the bundled SHMEM and the explicitly configured CATLASS installation into this workflow.
    for name in ("SHMEM_ROOT", "PYTHONPATH", "PYTHONHOME", "PYTHON_EXEC", "CONDA_PREFIX"):
        env.pop(name, None)
    prefix = Path(config["python"]).parent.parent
    env["PATH"] = str(Path(config["python"]).parent) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    # Pass paths as positional arguments, never interpolate configuration into shell code.
    probe = (
        "import json, os, pathlib, sys; "
        "env = dict(os.environ); "
        "env.update(CONDA_PREFIX=sys.prefix) if (pathlib.Path(sys.prefix) / 'conda-meta').is_dir() else None; "
        "print(json.dumps(env))"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >&2 || exit $?; exec "$2" -I -c "$3"',
            "bs32-env",
            config["cann_env"],
            config["python"],
            probe,
        ],
        env=env,
        cwd="/tmp",
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    env = json.loads(result.stdout)
    env.update(
        PYTHON=config["python"],
        DEEPEP_BUILD_MEGAMOE="ON",
        DEEPEP_MEGAMOE_TIMER="OFF",
        MASTER_ADDR=config["master_addr"],
        MASTER_PORT=str(config["master_port"]),
        DEEPEP_SHMEM_ENDPOINT=f"tcp://{config['master_addr']}:{config['shmem_port']}",
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, range(16))),
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONSAFEPATH="1",
    )
    env["LD_LIBRARY_PATH"] = f"{ROOT}/third-party/shmem/install/shmem/lib:{prefix}/lib:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    for name in ("SHMEM_ROOT", "PYTHONPATH", "PYTHONHOME", "PYTHON_EXEC"):
        env.pop(name, None)
    return env


class ErrorOutput:
    """Show diagnostic context without dumping every compiler warning to the terminal."""

    error = re.compile(
        r"\b(?:fatal\s+)?error\s*:|\berror\s+\d+\b|\bundefined (?:reference|symbol)\b|"
        r"\bkilled signal\b|^\s*(?:\[ERROR\]|CMake Error\b|Traceback\b|FAILED:)",
        re.IGNORECASE,
    )
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self):
        self.before = deque(maxlen=6)
        self.after = 0

    def write(self, line):
        plain = self.ansi.sub("", line)
        if self.error.search(plain):
            for previous in self.before:
                print(previous, end="", flush=True)
            self.before.clear()
            print(line, end="", flush=True)
            self.after = 8
        elif self.after:
            print(line, end="", flush=True)
            self.after -= 1
        elif plain.lstrip().startswith("[BUILD]"):
            print(line, end="", flush=True)
            self.before.clear()
        else:
            self.before.append(line)


def execute(command, env, log=None, cwd="/tmp", *, errors_only=False):
    print("[EXEC] " + shlex.join(map(str, command)), flush=True)
    if log is None:
        subprocess.run(command, env=env, cwd=cwd, check=True)
        return
    display = ErrorOutput() if errors_only else None
    with Path(log).open("w", encoding="utf-8") as stream:
        with subprocess.Popen(
            command, env=env, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if display is None:
                    print(line, end="", flush=True)
                else:
                    display.write(line)
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def preflight(config, env, *, max_bs=32):
    library = ROOT / "third-party/shmem/install/shmem/lib/libshmem.so"
    if not library.is_file():
        raise RuntimeError(f"Missing {library}; run 'bash experiments/megamoe/scripts/run_bs32.sh build' first")
    code = """import json, sys
from pathlib import Path
from importlib.resources import files
import torch, torch_npu, deep_ep
from deep_ep._native import require_native
import deep_ep_experimental.megamoe._ep32_C as megamoe
require_native('BS32 preflight')
if getattr(megamoe, 'api_protocol', None) != 17:
    raise RuntimeError('Installed MegaMoE protocol is outdated; run run_bs32.sh build on both hosts first')
info = json.loads(files('deep_ep').joinpath('_build_info.json').read_text())
if Path(deep_ep.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()):
    raise RuntimeError('deep_ep resolves to source; run build first')
if not info['megamoe']['enabled'] or info['shmem']['rdma_backend'] != sys.argv[2]:
    raise RuntimeError('Installed wheel does not match MegaMoE/RoCE configuration; run build first')
if sys.argv[3] == 'ON' and not megamoe.timer_enabled:
    raise RuntimeError('Trace requires a timer-enabled wheel; run run_bs32.sh build --timer on both hosts first')
if int(sys.argv[4]) > getattr(megamoe, 'hierarchical_max_tokens', 32):
    raise RuntimeError('Installed wheel does not support requested BS; run run_bs32.sh build on both hosts first')
if torch.npu.device_count() != 16:
    raise RuntimeError('BS32 requires sixteen visible NPUs on each host')
print('[CHECK] SHMEM:', [line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libshmem.so' in line][:1], 'MegaMoE:', getattr(megamoe, '__file__', None), flush=True)
print('[CHECK] Python:', sys.executable, 'deep_ep:', deep_ep.__file__, flush=True)
"""
    execute(
        [
            config["python"],
            "-I",
            "-c",
            code,
            str(ROOT),
            config["rdma_backend"],
            env["DEEPEP_MEGAMOE_TIMER"],
            str(max_bs),
        ],
        env,
    )


def test_commands(
    config,
    run_dir,
    *,
    mode="accuracy",
    timer_ranks=None,
    perf_method="legacy",
):
    python = [config["python"], "-I", "-m"]
    golden_dir = str(Path(config["case_dir"]) / "golden")
    common = [
        "--config",
        str(run_dir / "case.json"),
        "--input",
        "fp8",
        "--routing",
        "balanced",
        "--check-tokens",
        "0",
        "--golden-dir",
        golden_dir,
    ]
    rank = config["node_rank"]
    golden = [
        *python,
        "deep_ep_experimental.megamoe.golden",
        *common,
        "--ep-size",
        "32",
        "--ranks",
        f"{rank * 16}-{rank * 16 + 15}",
        "--workers",
        str(config["golden_workers"]),
    ]
    benchmark = [
        *python,
        "torch.distributed.run",
        "--nnodes",
        "2",
        "--nproc-per-node",
        "16",
        "--node-rank",
        str(rank),
        "--master-addr",
        config["master_addr"],
        "--master-port",
        str(config["master_port"]),
        "--rdzv-conf",
        "timeout=7200",
        "--max-restarts",
        "0",
        "--module",
        "deep_ep_experimental.megamoe.benchmark",
        *common,
        "--mode",
        mode,
        "--output-dir",
        str(run_dir),
    ]
    if timer_ranks is not None:
        benchmark.extend(["--timer", "--timer-ranks", ",".join(map(str, timer_ranks))])
    if mode in ("perf", "trace"):
        benchmark.extend(["--perf-method", perf_method])
    return golden, benchmark


def parse_timer_ranks(value):
    ranks = []
    try:
        for part in value.split(","):
            bounds = part.split("-")
            if len(bounds) == 1:
                ranks.append(int(part))
            elif len(bounds) == 2:
                first, last = map(int, bounds)
                if not 0 <= first <= last < 32:
                    raise ValueError
                ranks.extend(range(first, last + 1))
            else:
                raise ValueError
        if not ranks or min(ranks) < 0 or max(ranks) >= 32 or len(set(ranks)) != len(ranks):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use distinct EP32 ranks, e.g. 0,16 or 0-31") from error
    return sorted(ranks)


def performance_summary(config, run_dir, bs_values=(32,)):
    # All-rank statistics are replicated on every rank; each host reads its own disk.
    rank = config["node_rank"] * 16
    summaries, rows = [], []
    for bs in bs_values:
        case_dir = run_dir / f"bs_{bs}" if len(bs_values) > 1 else run_dir
        path = case_dir / f"rank_{rank}" / f"bs_{bs}" / "profiling.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        lines = [
            f"[PERF SUMMARY] EP32 BS{bs}; FP8 input / FP4 weights; balanced routing",
            f"Performance method: {report.get('perf_method', 'legacy')}",
            f"Implementation: {report['implementation']} (QP8 / Dispatch14 / cross-expert ring)",
            f"All-rank kernel Duration; {report['rounds']}/{report['recorded_rounds']} rounds (tail half)",
            f"Mean of per-round minimum rank: {report['mean_us']:.3f} us",
            f"Mean of per-round maximum rank: {report['rank_max_mean_us']:.3f} us",
            "These are rank-local kernel durations, not a cross-device timestamp span or Python wall time.",
            f"All-rank statistics: {path}",
            f"Local raw samples/profiler: {case_dir}/rank_<rank>/bs_{bs}/",
        ]
        summaries.append("\n".join(lines) + "\n")
        rows.append(
            dict(
                bs=bs,
                perf_method=report.get("perf_method", "legacy"),
                implementation=report["implementation"],
                rounds=report["rounds"],
                recorded_rounds=report["recorded_rounds"],
                mean_us=report["mean_us"],
                rank_max_mean_us=report["rank_max_mean_us"],
            )
        )
    summary = "\n".join(summaries)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary, end="", flush=True)


def trace_summary(run_dir, case, local_timer_ranks):
    summary = (
        "\n".join(
            [
                f"[TRACE SUMMARY] EP32 BS{case['bs_values'][0]}; FP8 input / FP4 weights; balanced routing",
                f"Warmup calls: {case['warmup']}; trace batch calls: {case['iters']}; only the final call is instrumented.",
                "Performance profiler disabled; run the perf action for a performance baseline.",
                f"Local traced ranks: {local_timer_ranks}",
                f"Local timer files: {run_dir / ('bs_' + str(case['bs_values'][0])) / 'timer'}",
                "Trace time is relative to each rank; merged ranks are not globally clock-aligned.",
            ]
        )
        + "\n"
    )
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="", flush=True)


def print_trace_parts(source, config, *, env=None):
    execute(
        [
            config["python"],
            "-I",
            str(ROOT / "experiments/megamoe/scripts/trace_text.py"),
            "export",
            str(source),
            "--max-bytes",
            "65535",
            "--print-parts",
        ],
        environment(config) if env is None else env,
        Path(source) / "trace_text.log",
    )


def dispatch(
    action,
    config,
    *,
    rebuild_shmem=False,
    errors_only=False,
    timer=False,
    warmup=None,
    iters=None,
    timer_ranks=None,
    include_wait=False,
    perf_method="legacy",
    extra_bs=False,
    bs=None,
    print_text=True,
):
    if bs is not None and (
        type(bs) is not int or not 1 <= bs <= 256 or action not in ("run", "golden", "perf", "trace")
    ):
        raise ValueError("--bs requires run/golden/perf/trace and an integer in [1,256]")
    if bs is not None and extra_bs:
        raise ValueError("--bs and --extra-bs are mutually exclusive")
    if extra_bs and action != "perf":
        raise ValueError("--extra-bs requires perf")
    if platform.system() != "Linux":
        raise RuntimeError("Build/run require Linux; configure/show-config can run without devices")
    env = environment(config)
    env["DEEPEP_MEGAMOE_TIMER"] = "ON" if timer or action == "trace" else "OFF"
    if action == "build":
        command = ["bash", str(ROOT / "experiments/megamoe/scripts/build.sh"), "--rdma-backend", config["rdma_backend"]]
        if rebuild_shmem:
            command.append("--rebuild-shmem")
        directory = Path(config["case_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        log_dir = Path(tempfile.mkdtemp(prefix=f"build_node_{config['node_rank']}_", dir=directory))
        log = log_dir / "build.log"
        print(f"[BUILD] Full log: {log}", flush=True)
        # Let pip stream native compiler diagnostics, rather than buffering them until failure.
        env.setdefault("PIP_VERBOSE", "1")
        try:
            execute(command, env, log, cwd=ROOT, errors_only=errors_only)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Build failed; full log: {log}", file=sys.stderr, flush=True)
            raise
        print(f"[DONE] Build/install completed; full log: {log}", flush=True)
        return
    if extra_bs:
        preflight(config, env, max_bs=max(EXTRA_PERF_BATCHES))
    elif bs is not None:
        preflight(config, env, max_bs=bs)
    else:
        preflight(config, env)
    directory = Path(config["case_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    sampling = action in ("perf", "trace")
    prefix = f"{action}_node_" if sampling else "node_"
    run_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}{config['node_rank']}_", dir=directory))
    case = dict(CASE)
    if bs is not None:
        case["bs_values"] = [bs]
    if extra_bs:
        case["bs_values"] = [32, *EXTRA_PERF_BATCHES]
    if sampling:
        default_warmup, default_iters = (2, 1) if action == "trace" else (20, 100)
        case.update(
            warmup=default_warmup if warmup is None else warmup,
            iters=default_iters if iters is None else iters,
        )
    selected = ([0, 16] if timer_ranks is None else timer_ranks) if action == "trace" else None
    (run_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    (run_dir / "launch.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            dict(
                action=action,
                timer_ranks=selected,
                warmup=case["warmup"],
                iters=case["iters"],
                include_wait=include_wait,
                perf_method=perf_method,
                extra_bs=extra_bs,
                implementation="dispatch_qp8_combine_qp16",
                bs_values=case["bs_values"],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[RUN] node={config['node_rank']} master={config['master_addr']} output={run_dir}", flush=True)
    if action == "perf":
        # Each BS gets its own process group and exact-size symmetric buffer.
        # Adding BS256 must not change BS32's capacity, layout or reset work.
        for bs in case["bs_values"]:
            case_dir = run_dir / f"bs_{bs}" if extra_bs else run_dir
            case_dir.mkdir(exist_ok=True)
            single_case = dict(case, bs_values=[bs])
            (case_dir / "case.json").write_text(json.dumps(single_case, indent=2) + "\n", encoding="utf-8")
            _, benchmark = test_commands(
                config,
                case_dir,
                mode="perf",
                perf_method=perf_method,
            )
            print(f"[PERF] EP32 BS={bs} output={case_dir}", flush=True)
            execute(benchmark, env, case_dir / "console.log")
        performance_summary(config, run_dir, case["bs_values"])
        print(f"[DONE] {action}: {run_dir}", flush=True)
        return run_dir
    golden, benchmark = test_commands(
        config,
        run_dir,
        mode=action if sampling else "accuracy",
        timer_ranks=selected,
        perf_method=perf_method,
    )
    if not sampling:
        execute(golden, env, run_dir / "golden.log")
    if action != "golden":
        execute(benchmark, env, run_dir / "console.log")
    local_ranks = [rank for rank in selected or [] if rank // 16 == config["node_rank"]]
    if local_ranks:
        execute(
            [
                config["python"],
                "-I",
                "-m",
                "deep_ep_experimental.megamoe.timer",
                str(run_dir / f"bs_{case['bs_values'][0]}" / "timer"),
                "--ranks",
                ",".join(map(str, local_ranks)),
                *(["--include-wait"] if include_wait else []),
                "--merge",
            ],
            env,
            run_dir / "parse_timer.log",
        )
    if action == "trace":
        trace_summary(run_dir, case, local_ranks)
        if print_text and local_ranks:
            print_trace_parts(run_dir, config, env=env)
    print(f"[DONE] {action}: {run_dir}", flush=True)
    return run_dir


def main(argv=None, *, print_text=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=("configure", "show-config", "build", "golden", "run", "perf", "trace"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Machine-local JSON settings file")
    parser.add_argument(
        "--rebuild-shmem", action="store_true", help="build only: rebuild the bundled SDK after stopping its users"
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="build only: show errors with nearby context; retain all output in build.log",
    )
    parser.add_argument(
        "--extra-bs",
        action="store_true",
        help="perf only: after BS32, also measure BS1,4,8,16,256 with identical performance options",
    )
    parser.add_argument("--bs", type=int, help="run/golden/perf/trace: tokens per rank in [1,256] (default: 32)")
    parser.add_argument("--timer", action="store_true", help="build only: enable device timer support for trace")
    waits = parser.add_mutually_exclusive_group()
    waits.add_argument(
        "--include-wait",
        dest="include_wait",
        action="store_true",
        default=None,
        help="trace only: show WAIT events, including RDMA completion/receive waits (default: hidden)",
    )
    waits.add_argument(
        "--hide-wait",
        dest="include_wait",
        action="store_false",
        help="trace only: hide WAIT events without changing timestamps or raw samples",
    )
    parser.add_argument("--warmup", type=int, help="perf/trace only: warmup calls (default: perf 20, trace 2)")
    parser.add_argument(
        "--perf-method",
        choices=("legacy", "legacy-read", "legacy-zero", "shmem"),
        default=None,
        help="perf/trace only: legacy Exp20 batch (default) or per-round SHMEM alignment",
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="perf/trace only: batch calls (default: perf 100, report last half; trace 1, instrument final call)",
    )
    parser.add_argument(
        "--timer-ranks", type=parse_timer_ranks, help="trace only: selected global ranks (default: 0,16; all: 0-31)"
    )
    for name in FIELDS:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int if name in ("node_rank", "master_port", "shmem_port", "golden_workers") else str,
            help="configure only: persist this setting",
        )
    args = parser.parse_args(argv)
    if args.bs is not None:
        if args.action not in ("run", "golden", "perf", "trace") or not 1 <= args.bs <= 256:
            parser.error("--bs requires run/golden/perf/trace and an integer in [1,256]")
        if args.extra_bs:
            parser.error("--bs and --extra-bs are mutually exclusive")
    if args.action != "configure" and any(getattr(args, name) is not None for name in FIELDS):
        parser.error("Settings may only be changed with configure")
    if args.rebuild_shmem and args.action != "build":
        parser.error("--rebuild-shmem requires build")
    if args.errors_only and args.action != "build":
        parser.error("--errors-only requires build")
    if args.extra_bs and args.action != "perf":
        parser.error("--extra-bs requires perf")
    if args.timer and args.action != "build":
        parser.error("--timer requires build; use the trace action for collection")
    if args.timer_ranks is not None and args.action != "trace":
        parser.error("--timer-ranks requires trace")
    if args.perf_method is not None and args.action not in ("perf", "trace"):
        parser.error("--perf-method requires perf or trace")
    if args.include_wait is not None and args.action != "trace":
        parser.error("--include-wait/--hide-wait require trace")
    if args.warmup is not None or args.iters is not None:
        if args.action not in ("perf", "trace"):
            parser.error("--warmup/--iters require perf or trace")
        if (args.warmup is not None and args.warmup < 0) or (args.iters is not None and args.iters < 1):
            parser.error("--warmup must be >= 0 and --iters must be >= 1")
    path = args.config.expanduser().absolute()
    if args.action == "configure":
        configure(path, args)
        return
    if not path.is_file():
        raise ValueError(
            f"No saved settings at {path}; run 'bash experiments/megamoe/scripts/run_bs32.sh configure' once on each host"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    validate(config)
    print(f"[CONFIG] Using: {path}", flush=True)
    if args.action == "show-config":
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    options = dict(rebuild_shmem=args.rebuild_shmem, errors_only=args.errors_only)
    if args.bs is not None:
        options["bs"] = args.bs
    if args.extra_bs:
        options["extra_bs"] = True
    if args.timer:
        options["timer"] = True
    if args.action in ("perf", "trace"):
        options.update(warmup=args.warmup, iters=args.iters, timer_ranks=args.timer_ranks)
        if args.perf_method is not None:
            options["perf_method"] = args.perf_method
    if args.action == "trace":
        options["include_wait"] = bool(args.include_wait)
        options["print_text"] = print_text
    return dispatch(args.action, config, **options)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
