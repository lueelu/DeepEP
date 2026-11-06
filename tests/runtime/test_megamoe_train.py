"""Training owner and package selection contracts; no device qualification."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from deep_ep import _runtime


class TrainingRuntime(_runtime.Runtime):
    _megamoe_train = True


def test_training_owner_rejects_second_owner_and_releases(monkeypatch):
    calls = []
    session = SimpleNamespace(close=lambda: calls.append("close"), barrier=lambda: calls.append("barrier"))
    native = SimpleNamespace(
        Runtime=lambda *args: pytest.fail("selected core factory"),
        create_megamoe_train_runtime=lambda *args: calls.append(args) or session,
        runtime_busy=lambda: False,
    )
    npu = SimpleNamespace(
        current_device=lambda: 0,
        Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(_runtime, "require_native", lambda operation: native)
    monkeypatch.setattr(_runtime, "npu_api", lambda: npu)
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setenv("DEEPEP_SHMEM_ENDPOINT", "tcp://127.0.0.1:29862")
    runtime = TrainingRuntime(None, 3 << 30, rank=767, world_size=768)
    assert calls == [(767, 768, 3 << 30, "tcp://127.0.0.1:29862", 120)]
    assert _runtime._OWNER is runtime
    with pytest.raises(RuntimeError, match="already owns"):
        TrainingRuntime(None, 2 << 20, rank=0, world_size=2)
    runtime.close()
    runtime.close()
    assert calls[-1] == "close" and calls.count("close") == 1
    assert _runtime._OWNER is None


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_optional_package_combinations(monkeypatch, training, inference, fused):
    import setuptools

    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    for key, value in (
        ("DEEPEP_BUILD_MEGAMOE_TRAIN", training),
        ("DEEPEP_BUILD_MEGAMOE", inference),
        ("DEEPEP_BUILD_EP_GMM_FUSED", fused),
    ):
        monkeypatch.setenv(key, "ON" if value else "OFF")
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(root / "setup.py"))
    for component, selected in (("megamoe_train", training), ("megamoe", inference), ("ep_gmm_fused", fused)):
        package = f"deep_ep_experimental.{component}"
        assert (package in captured["packages"]) is selected
        if selected:
            path = captured["package_dir"].get(package)
            if path is None:
                path = captured["package_dir"]["deep_ep_experimental"] + "/" + component
            assert (root / path / "__init__.py").is_file()
    assert len(captured["packages"]) == len(set(captured["packages"]))


def test_training_close_drains_stream_and_is_idempotent():
    # Exercise the actual wrapper's close method without importing Torch/NPU.
    import ast

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegaMoeBF16Train")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "close")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    calls = []
    op = SimpleNamespace(
        _closed=False,
        _stream=SimpleNamespace(synchronize=lambda: calls.append("sync")),
        _runtime=SimpleNamespace(ensure_alive=lambda: None, close=lambda: calls.append("session-close")),
        _executor=SimpleNamespace(close=lambda: calls.append("op-close")),
    )
    close = namespace["close"]
    close(op)
    close(op)
    assert calls == ["sync", "session-close", "op-close"] and op._closed


@pytest.fixture
def topk_dtype_check():
    """Exercise the actual source method body without importing Torch/NPU."""
    import ast

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_TrainRuntime")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "validate_topk_weight_type"
    )

    class Group:
        peer = None

        def __init__(self):
            self.calls = []

        def all_gather(self, value, stage):
            self.calls.append((value, stage))
            return [value, self.peer if self.peer is not None else value]

    namespace = {"torch": SimpleNamespace(bfloat16="bf16", float32="fp32"), "StoreGroup": Group}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["validate_topk_weight_type"], Group


@pytest.mark.parametrize("debug", [False, True])
def test_topk_local_dtype_check_always_rejects_unsupported_dtype(topk_dtype_check, debug):
    check, group_type = topk_dtype_check
    group = group_type()
    runtime = SimpleNamespace(group=group, debug_collectives=debug)
    with pytest.raises(ValueError, match="bfloat16 or float32"):
        check(runtime, "fp16", 0)
    if not debug:
        assert not group.calls


@pytest.mark.parametrize("debug", [False, True])
def test_mixed_queued_dtype_launches_do_not_require_one_persistent_dtype(topk_dtype_check, debug):
    check, group_type = topk_dtype_check
    group = group_type()
    runtime = SimpleNamespace(group=group, debug_collectives=debug)
    # Different forwards may retain their own p dtype until reverse backwards.
    # This is dtype/stage agreement, not a graph-identity or full order protocol.
    for stage, dtype in ((0, "bf16"), (0, "fp32"), (1, "fp32"), (1, "bf16")):
        check(runtime, dtype, stage)
    assert len(group.calls) == (4 if debug else 0)


@pytest.mark.parametrize("peer", [[0, 1], [1, 0]])
def test_debug_topk_coordination_rejects_peer_dtype_or_stage(topk_dtype_check, peer):
    check, group_type = topk_dtype_check
    group = group_type()
    runtime = SimpleNamespace(group=group, debug_collectives=True)
    group.peer = peer
    with pytest.raises(ValueError, match="EP ranks"):
        check(runtime, "bf16", 0)
    assert len(group.calls) == 1


@pytest.mark.parametrize("debug", [False, True])
def test_training_wrapper_passes_debug_option_to_owning_runtime(monkeypatch, debug):
    """Exercise the real constructor's default and explicit debug forwarding."""
    import ast
    import sys
    from types import ModuleType

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegaMoeBF16Train")
    calls = []
    session = SimpleNamespace(native=object(), device=0, rank=0, world_size=2, barrier=lambda: None)

    def runtime(*args, **kwargs):
        calls.append((args, kwargs))
        return session

    class Module:
        pass

    package = ModuleType("training_wrapper_projection")
    package.__path__ = []
    loaded = []

    def require_training_native(operation):
        loaded.append(operation)
        return SimpleNamespace(megamoe_train_workspace_address=lambda native: 1234)

    def extension_attribute(name):
        if name != "_C":
            raise AttributeError(name)
        assert loaded == ["MegaMoeBF16Train"], "Root loader must preload the bundled SHMEM closure"
        return SimpleNamespace(BF16=lambda *args: object())

    package.__getattr__ = extension_attribute
    monkeypatch.setitem(sys.modules, package.__name__, package)
    namespace = {
        "__name__": package.__name__ + ".bf16",
        "__package__": package.__name__,
        "torch": SimpleNamespace(nn=SimpleNamespace(Module=Module), empty=lambda *a, **kw: object(), bfloat16="bf16"),
        "npu_api": lambda: SimpleNamespace(current_stream=lambda: object()),
        "require_native": require_training_native,
        "_TrainRuntime": runtime,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    group = object()
    kwargs = {"debug_collectives": True} if debug else {}
    namespace["MegaMoeBF16Train"](group, 2 << 20, **kwargs)
    assert len(calls) == 1
    assert calls[0][1]["debug_collectives"] is debug


def test_precision_worker_bootstraps_root_before_training_extension(monkeypatch):
    """Execute the real worker setup without a framework, device or native DLL."""
    import ast
    from contextlib import nullcontext
    from datetime import timedelta
    import sys
    from types import ModuleType
    from urllib.parse import urlsplit

    import deep_ep._native as root_native

    source = Path(__file__).resolve().parents[2] / "experiments/megamoe_train/tests/bf16/test_mega_moe.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_worker")
    events = []
    torch = ModuleType("torch")
    torch.__path__ = []
    torch.npu = SimpleNamespace(set_device=lambda device: events.append("device"))
    distributed = ModuleType("torch.distributed")
    distributed.TCPStore = lambda *args, **kwargs: object()
    distributed.init_process_group = lambda *args, **kwargs: events.append("group")
    torch.distributed = distributed
    package = ModuleType("deep_ep_experimental")
    package.__path__ = []
    training = ModuleType("deep_ep_experimental.megamoe_train")
    operator = object()
    training.MegaMoeBF16Train = operator
    for name, module in {
        "torch": torch,
        "torch.distributed": distributed,
        "torch_npu": ModuleType("torch_npu"),
        "deep_ep_experimental": package,
        "deep_ep_experimental.megamoe_train": training,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def root_loader(operation):
        events.append("root-loader")
        return object()

    def training_import(name):
        assert name == "deep_ep_experimental.megamoe_train._C"
        assert "root-loader" in events, "Training DLL import bypassed the bundled SHMEM loader"
        events.append("training-extension")
        return object()

    monkeypatch.setattr(root_native, "require_native", root_loader)
    namespace = {
        "phase_timer": lambda *args: lambda stage: nullcontext(),
        "log_visible": lambda rank: False,
        "installed_root": lambda: None,
        "importlib": SimpleNamespace(import_module=training_import),
        "urlsplit": urlsplit,
        "timedelta": timedelta,
    }
    exec(compile(ast.Module(body=[worker], type_ignores=[]), str(source), "exec"), namespace)
    args = SimpleNamespace(devices=[3, 5], shmem_ip_port="tcp://127.0.0.1:29862", timeout=120)
    assert namespace["prepare_worker"](0, args, 0, 2, {}) is operator
    assert events == ["device", "root-loader", "training-extension", "group"]


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("peer_mismatch", [False, True])
def test_training_bootstrap_agrees_on_debug_before_creating_native_owner(monkeypatch, debug, peer_mismatch):
    """Training uses the common one-time configuration gate, even with debug off."""

    class Group:
        def rank(self):
            return 0

        def size(self):
            return 2

        def all_gather(self, value, stage):
            assert stage == "runtime configuration"
            peer = list(value)
            peer[6] = 1  # A distinct local NPU on the same mocked Linux host.
            if peer_mismatch:
                peer[-1] ^= 16  # The shared runtime wire bit for debug_collectives.
            return [value, peer]

    created = []
    native = SimpleNamespace(
        Runtime=lambda *args: pytest.fail("selected core factory"),
        create_megamoe_train_runtime=lambda *args: created.append(args) or SimpleNamespace(close=lambda: None),
        runtime_busy=lambda: False,
    )
    monkeypatch.setattr(_runtime, "StoreGroup", Group)
    monkeypatch.setattr(_runtime, "require_native", lambda operation: native)
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setattr(_runtime, "_host_boot_id", lambda: "same-host")
    monkeypatch.setattr(
        _runtime,
        "npu_api",
        lambda: SimpleNamespace(
            current_device=lambda: 0, Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: None)
        ),
    )
    monkeypatch.setenv("DEEPEP_SHMEM_ENDPOINT", "tcp://127.0.0.1:29862")
    if peer_mismatch:
        with pytest.raises(ValueError, match="debug_collectives"):
            TrainingRuntime(Group(), 2 << 20, debug_collectives=debug)
        assert not created and _runtime._OWNER is None
    else:
        runtime = TrainingRuntime(Group(), 2 << 20, debug_collectives=debug)
        assert runtime.debug_collectives is debug and len(created) == 1
        runtime.close()


def test_expert_ready_builds_use_distinct_cache_and_record_actual_policy(monkeypatch, tmp_path):
    import json
    import sys
    from types import ModuleType

    import build_support as build
    from setuptools import Distribution, Extension

    pybind11 = ModuleType("pybind11")
    pybind11.get_cmake_dir = lambda: str(tmp_path / "pybind11")
    monkeypatch.setitem(sys.modules, "pybind11", pybind11)
    monkeypatch.setenv("DEEPEP_BUILD_EP_GMM_FUSED", "OFF")
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    # Exercise the real build command and cache key without invoking SDK tools.
    configuration = {
        "megamoe_train": {"enabled": True, "force_expert_ready": False},
        "source_sha256": "current",
        "source_version": "1",
        "cmake_path": "cmake",
        "compiler_path": "c++",
        "ASCEND_HOME_PATH": "/sdk/cann",
        "SHMEM_ROOT": "/sdk/shmem",
        "CATLASS_ROOT": "/sdk/catlass",
        "build_type": "Release",
    }
    monkeypatch.setattr(build, "build_configuration", lambda: configuration)
    configure_calls = []

    def run(argv):
        argv = list(map(str, argv))
        if "-B" in argv:
            configure_calls.append(argv)
            directory = Path(argv[argv.index("-B") + 1])
            (directory / "compiler-info.json").write_text('{"id":"test"}', encoding="utf-8")
            (directory / "capabilities.json").write_text('{"capabilities":[]}', encoding="utf-8")

    monkeypatch.setattr(build, "checked_run", run)
    monkeypatch.setattr(build, "bundle_shmem_runtime", lambda sdk, target: {"test": "hash"})
    published = []
    monkeypatch.setattr(build, "publish_source_native", published.append)
    ext = Extension("deep_ep._C", sources=[])
    for forced in (False, True):
        configuration["megamoe_train"]["force_expert_ready"] = forced
        command = build.CMakeBuild(Distribution({"ext_modules": [ext]}))
        command.ensure_finalized()
        command.build_temp = str(tmp_path / "temp")
        command.build_lib = str(tmp_path / "lib")
        destination = Path(command.get_ext_fullpath(ext.name)).parent
        destination.mkdir(parents=True, exist_ok=True)
        command.build_extension(ext)
        args = configure_calls[-1]
        assert f"-DDEEPEP_BF16_FORCE_EXPERT_READY={'ON' if forced else 'OFF'}" in args
        assert "-DDEEPEP_BUILD_MEGAMOE_TRAIN=ON" in args
        info = json.loads((destination / "_build_info.json").read_text(encoding="utf-8"))
        assert info["megamoe_train"]["force_expert_ready"] is forced
        assert info["shmem_runtime"] == {"test": "hash"}
        assert published[-1] == destination.resolve()
    object_directories = [args[args.index("-B") + 1] for args in configure_calls]
    assert len(set(object_directories)) == 2
    assert len(published) == 2
