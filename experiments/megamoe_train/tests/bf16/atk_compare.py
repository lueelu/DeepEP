# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""ATK CV double-benchmark comparison of every rank/output.

Use ATK's registered implementation, without flattening different gradients or
ranks together. The existing launcher owns execution; ATK owns the double-benchmark
precision decision. No replacement formulas or fallbacks are used.
"""

from importlib import metadata
import os
from pathlib import Path

import torch

from precision_metrics import OUTPUTS
from npu_baseline import BIGOP_REFERENCE_VERSION


STANDARD = "cv_fused_double_benchmark"
DEFAULT_RATIOS = (5.0, 1.5, 1.5)


def require_atk():
    try:
        from atk.tasks.post_process import ACCURACY_REGISTRY
        from atk.tasks.post_process.private_loader import (
            get_accuracy_provider,
            get_private_accuracy_load_error,
            is_private_accuracy_loaded,
        )
    except ImportError as error:
        raise RuntimeError("Precision requires ATK and atk-private-precision in this Python environment") from error
    error = get_private_accuracy_load_error()
    if error is not None or not is_private_accuracy_loaded() or STANDARD not in ACCURACY_REGISTRY:
        raise RuntimeError(f"ATK private comparator {STANDARD} is unavailable: {error!r}")
    comparator = ACCURACY_REGISTRY[STANDARD]
    if get_accuracy_provider(STANDARD) != "private" or not comparator.__module__.startswith("atk_private_precision."):
        raise RuntimeError(f"ATK {STANDARD} must come from atk-private-precision")
    return comparator, {
        "standard": STANDARD,
        "provider": "private",
        "atk_version": metadata.version("ATK"),
        "private_version": metadata.version("atk-private-precision"),
        "implementation": comparator.__module__ + "." + comparator.__name__,
    }


def compare_with_atk(actual, baseline, truth, directory, ratios, *, outputs=OUTPUTS):
    """ATK local=actual, remote=FP32-boundary BigOp, bm_path=CPU FP64 golden.

    This is CompareExecutor's three-role contract, called separately for the
    five files in one rank, or the selected cached outputs. Keep the golden
    tensor in FP64 and actual/benchmark in their original output dtype:
    BF16, except dtopk_weights follows the BF16/FP32 route-weight input.
    Retain failing triples for later diagnosis.
    """
    outputs = tuple(outputs)
    if not outputs or len(set(outputs)) != len(outputs) or any(name not in OUTPUTS for name in outputs):
        raise ValueError(f"Expected distinct output names from {OUTPUTS}, got {outputs}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["ATK_TASK_OUTPUT_PATH"] = str(directory)
    comparator_type, identity = require_atk()
    from atk.common.connection import RemoteManager
    from atk.configs.base_config import TaskType
    from atk.configs.case_config import CaseConfig
    from atk.configs.nodes_config import Node, NodesConfig
    from atk.configs.results_config import TaskResult
    from atk.tasks.executors.compare_excutor import CompareExecutor

    thresholds = dict(zip(("max_re_ratio", "avg_re_ratio", "root_mean_squared_ratio"), ratios, strict=True))
    thresholds["number_count_radio"] = 2
    case = CaseConfig(id=0, name="megamoe_bf16_train", inputs=[], standard={"acc": {STANDARD: thresholds}})
    main_node = Node(backend="npu", name="actual", task=[TaskType.ACCURACY])
    baseline_node = Node(backend="npu", name="bigop", task=[TaskType.ACCURACY])
    task = TaskResult(
        case_config=case,
        nodes=NodesConfig(nodes=[main_node, baseline_node]),
        task_type=[TaskType.ACCURACY],
        backend="npu",
        benchmark_device="cpu",
    )
    executor = CompareExecutor(task)
    comparator = executor.compare
    if not isinstance(comparator, comparator_type):
        raise RuntimeError("ATK CompareExecutor did not select the registered private CV comparator")
    # Check that constructor kwargs reached the actual plugin configuration.
    for key, value in thresholds.items():
        if getattr(comparator.benchmark_standard, key) != value:
            raise RuntimeError(f"ATK did not apply configured {key}={value}")
    golden_dir = directory / "golden"
    golden_dir.mkdir()
    comparator.bm_path = str(golden_dir)
    # Inputs are already sharded by rank and copied to CPU. ATK's golden loader
    # still needs its normal local-node context, without a distributed rank_0
    # directory rewrite or an HTTP connection to another executor.
    comparator.remote_manager = RemoteManager(case, main_node, baseline_node)
    report = {
        **identity,
        "benchmark_reference_version": BIGOP_REFERENCE_VERSION,
        "compared_outputs": list(outputs),
        "thresholds": thresholds,
        "passed": True,
        "outputs": {},
    }
    for name in outputs:
        index = OUTPUTS.index(name)
        a, b, g = (group[name].detach().cpu().contiguous() for group in (actual, baseline, truth))
        if a.shape != b.shape or a.shape != g.shape or a.dtype != b.dtype or g.dtype != torch.float64:
            raise ValueError(f"ATK {name}: shapes must match, actual/BigOp dtypes must match, golden must be FP64")
        data_file = f"output_{index}.pt"
        golden_file = golden_dir / data_file
        if all(bool(torch.isfinite(value).all()) for value in (a, b, g)):
            torch.save(g.reshape(-1), golden_file)
            result = comparator.compute_accuracy_result(a.reshape(-1), b.reshape(-1), data_file)
            detail = result.model_dump()
            if type(detail.get("result")) is not bool:
                raise RuntimeError(f"ATK returned no boolean decision for {name}: {detail}")
            passed = detail["result"]
        else:
            passed, detail = False, {"result": False, "error_info": "Non-finite actual, BigOp or CPU FP64 golden"}
        record = {"passed": passed, "shape": list(a.shape), "dtype": str(a.dtype), "file": data_file, "result": detail}
        if passed:
            golden_file.unlink(missing_ok=True)
        else:
            torch.save(g, golden_file)
            for role, value in (("actual", a), ("benchmark", b)):
                target = directory / role
                target.mkdir(exist_ok=True)
                torch.save(value, target / data_file)
            record["artifacts"] = str(directory)
        report["outputs"][name] = record
        report["passed"] &= passed
    return report
