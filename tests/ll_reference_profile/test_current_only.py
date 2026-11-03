"""Reference-only production tracing and searchable word/document contracts."""
from pathlib import Path
import re

from deep_ep.utils.ll_reference_profile_parse import DATA_WORDS, STAGE_INFO, WORDS

ROOT = Path(__file__).resolve().parents[2]
LL = ROOT / 'csrc/kernels/dispatch/low_latency'


def test_no_legacy_device_tracers_or_transport_timer_parameters():
    for directory in (LL, ROOT / 'csrc/kernels/combine'):
        for path in directory.rglob('*'):
            if path.suffix not in ('.h', '.cpp', '.hpp') or path.name.endswith('_layout.h'):
                continue
            source = path.read_text(encoding='utf-8')
            assert not re.search(r'Ll(?:Stage|Expert)Timer|\btimer_|\bexpertTimer_', source), path
    for name in ('ll_stage_timer_device.h', 'll_expert_timer_device.h'):
        assert not (LL / 'common' / name).exists()
    cmake = (ROOT / 'cmake/device/CMakeLists.txt').read_text(encoding='utf-8')
    assert 'deepep_kernel_ll_timer' not in cmake
    assert 'if(DEEPEP_LL_STAGE_TIMER)' in cmake
    assert 'FATAL_ERROR "Legacy LL tracing was removed' in cmake


def test_all_profile_words_have_searchable_annotations_and_documented_formulas():
    seen = set()
    # The single entry translation unit owns w0/w1; every other word stays in LL headers.
    sources = [*LL.rglob('*.h'), ROOT / 'csrc/kernels/dispatch/low_latency.cpp']
    for path in sources:
        source = path.read_text(encoding='utf-8')
        for line in source.splitlines():
            match = re.search(r'(?:Mark|Value)<(\d+)>\(', line)
            if match:
                index = int(match[1])
                assert f'w{index}: {WORDS[index]}' in line, (path, line)
                seen.add(index)
    assert seen == DATA_WORDS
    document = (ROOT / 'tests/ll_rdma/STAGE_TIMELINE.md').read_text(encoding='utf-8')
    for name, info in STAGE_INFO.items():
        assert f'`{name}` / {info["label"]}' in document
        assert f'`{info["reference_formula"]}`' in document
    assert document.count('```mermaid') == 2


def test_one_click_rejects_retired_timer_before_preparation():
    import runpy
    import pytest
    launch = runpy.run_path(str(ROOT / 'tests/ll_rdma/one_click.py'))
    with pytest.raises(SystemExit):
        launch['parse_args'](['0', '192.0.2.1', 'old-mode', '--mode', 'timer'])
    args = launch['parse_args'](['0', '192.0.2.1', 'new-mode'])
    assert args.mode == 'reference'
    assert '--dispatch-only' in launch['test_arguments'](args)
