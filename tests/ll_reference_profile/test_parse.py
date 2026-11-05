"""Synthetic CPU contract checks; these fixtures are not device measurements."""
import copy
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timer = load('reference_parser_test', ROOT/'deep_ep/utils/ll_reference_profile_parse.py')
analyzer = load('reference_analyzer_test', Path(__file__).with_name('analyze.py'))


def sample(rank=0, iteration=0, cores=16, world=16, frequency=1e9, offset=10**17):
    raw = []
    scale = iteration + 1
    for core in range(cores):
        base = offset + rank*10000 + core*1000
        points = {0:0, 1:10, 2:12, 3:20, 4:30, 5:40, 6:41, 7:100,
                  20:170, 21:180, 57:181}
        counts = {}
        if core < cores//2:
            points.update({8:101, 9:160})
            counts[43] = int(core == 0)
            if counts[43]:
                points.update({38:120, 39:140, 40:159})
        else:
            points.update({10:101, 11:115, 12:116, 13:125, 14:126, 15:135,
                           16:136, 17:140, 18:141, 19:168, 24:102, 25:110,
                           26:117, 27:120, 28:127, 29:130, 30:145, 31:150})
            counts.update({37:2 if core == cores-1 else 0, 33:0, 34:0, 41:0})
            if counts[37]:
                points.update({32:155, 35:165, 36:166, 42:157})
                counts.update({33:2*scale, 34:4*scale, 41:2})
        words = [0]*timer.WORD_COUNT
        for index, value in points.items():
            words[index] = base + value*scale
        for index, value in counts.items():
            words[index] = value
        words[50:57] = [rank, core, world, cores, 1, timer.SCHEMA, 1]
        words[58] = sum(1 << index for index in (*points, *counts))
        words[59:62] = [timer.MAGIC, 1, 1]
        words[64] = 1
        if core >= cores//2:
            total=counts[37]
            words[96:108] = [0,0,total,total,total,1 if total else 0,
                             words[21],words[21],0,1,1,int(total>0)]
        raw.append([words[n:n+8] for n in range(0,timer.WORD_COUNT,8)])
    return dict(format=timer.FORMAT, raw=raw, operation='dispatch', frequency_hz=frequency,
                metadata=dict(rank=rank, world_size=world, iteration=iteration, correctness='passed',
                              run_id='synthetic-only', case_id=0, route='normal', seed=7,
                              profile_seed=7, route_sha256='1'*64,
                              config=dict(num_tokens=1, hidden=3584, dtype='fp8'),
                              sampling=dict(diagnostic_iters=2),
                              comparison=dict(source_revision='test-source', source_dirty=False,
                                              installed_source_sha256='2'*64)))


def getword(raw, core, index):
    return raw[core][index//8][index%8]


def setword(raw, core, index, value):
    raw[core][index//8][index%8] = value


def omitword(raw, core, index):
    setword(raw, core, index, 0)
    setword(raw, core, 58, getword(raw, core, 58) & ~(1 << index))


def inputs(directory, ranks=range(16), **kwargs):
    paths = []
    for rank in ranks:
        for iteration in range(2):
            folder = directory/'dispatch'/f'rank_{rank:03d}'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/f'sample_{iteration:04d}.json'
            path.write_text(json.dumps(sample(rank, iteration, **kwargs)), encoding='utf-8')
            paths.append(path)
    return paths


@pytest.mark.parametrize('cores', [16, 56, 72])
def test_dynamic_cores_exact_integer_subtraction_and_loop_accumulation(cores):
    saved = sample(cores=cores)
    report = timer.parse_records(saved['raw'])
    assert report['active_aiv'] == cores
    assert len(report['cores']) == cores
    kernel = next(r for r in report['records'] if r['core'] == 0 and r['stage'] == 'kernel')
    assert kernel['total_ticks'] == 180 and kernel['total_us'] == .18
    poll = next(r for r in report['records'] if r['core'] == cores-1 and r['stage'] == 'recv_packet_poll')
    assert poll['total_ticks'] == 2 and poll['calls'] == 2 and poll['first_tick'] is None
    assert timer.WORDS[57] == 'before_finalize'


def test_zero_timestamp_and_executed_zero_differ_from_absent():
    raw = sample(offset=0)['raw']
    setword(raw, 0, 1, 0)
    records = timer.parse_records(raw, frequency_hz=None)['records']
    first = next(r for r in records if r['core'] == 0 and r['stage'] == 'entry_config')
    assert first['total_ticks'] == 0 and 'total_us' not in first
    assert not any(r['core'] == 1 and r['stage'] == 'send_packets' for r in records)
    zero = next(r for r in records if r['core'] == 8 and r['stage'] == 'recv_packet_poll')
    assert zero['total_ticks'] == 0 and zero['calls'] == 0


@pytest.mark.parametrize('index,value', [(55,1),(56,2),(59,0),(60,2),(61,0),(62,1),(53,56),(50,16),(52,8),(22,1)])
def test_header_and_reserved_damage_rejected(index, value):
    raw = sample()['raw']; setword(raw,0,index,value)
    with pytest.raises(ValueError): timer.parse_records(raw)


def test_validity_shape_timestamp_and_frequency_validation():
    for index, value in [(58, getword(sample()['raw'],0,58) | 1 << 59), (39, 1), (21,0)]:
        raw = sample()['raw']; setword(raw,0,index,value)
        with pytest.raises(ValueError): timer.parse_records(raw)
    raw = sample()['raw']; setword(raw,1,38,123)
    with pytest.raises(ValueError, match='Unmarked'): timer.parse_records(raw)
    with pytest.raises(ValueError, match='shape'): timer.parse_records([c[:7] for c in sample()['raw']])
    with pytest.raises(ValueError, match='dispatch'): timer.parse_records(sample()['raw'], 'combine')
    for value in (0, -1, True, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='frequency'): timer.parse_records(sample()['raw'], frequency_hz=value)


def test_cpp_word_names_and_constants_match():
    source = (ROOT/'csrc/kernels/dispatch/low_latency/common/ll_reference_profile_layout.h').read_text()
    block = source.split('kNames[kWordsPerCore] = {',1)[1].split('};',1)[0]
    assert tuple(re.findall(r'"([^"]+)"', block)) == timer.WORDS
    assert 'kSchema = 4U' in source and 'kMagic = 0x4c4c4350ULL' in source


@pytest.mark.parametrize('world', [16, 32, 64, 128, 256])
def test_all_current_ll_domains_are_accepted(world):
    raw = sample(rank=world-1, world=world, cores=72)['raw']
    parsed = timer.parse_records(raw)
    assert parsed['rank'] == world-1 and parsed['world_size'] == world


@pytest.mark.parametrize('core,word',
    [(0, w) for w in (*range(8), 8, 9, 20, 21, 43, 57)] +
    [(8, w) for w in (*range(10, 20), 30, 31, 33, 34, 37, 41)])
def test_cleared_mandatory_point_is_rejected_even_with_complete_header(core, word):
    raw = sample()['raw']
    omitword(raw, core, word)
    with pytest.raises(ValueError, match='mandatory'):
        timer.parse_records(raw)


@pytest.mark.parametrize('core,word', [(0, 38), (0, 39), (0, 40),
                                       (15, 32), (15, 35), (15, 36), (15, 42)])
def test_nonempty_task_requires_its_complete_detail_points(core, word):
    raw = sample()['raw']
    omitword(raw, core, word)
    with pytest.raises(ValueError, match='Nonempty'):
        timer.parse_records(raw)


@pytest.mark.parametrize('core,word', [(1, 38), (1, 39), (1, 40),
                                       (8, 32), (8, 35), (8, 36), (8, 42)])
def test_empty_task_rejects_a_valid_task_only_timestamp(core, word):
    raw = sample()['raw']
    setword(raw, core, word, getword(raw, core, 0)+155)
    setword(raw, core, 58, getword(raw, core, 58) | (1 << word))
    with pytest.raises(ValueError, match='Empty'):
        timer.parse_records(raw)


@pytest.mark.parametrize('core,word,value', [(8, 33, 1), (8, 34, 1), (8, 41, 1),
                                              (15, 41, 0), (15, 41, 3)])
def test_receive_task_and_batch_counters_are_consistent(core, word, value):
    raw = sample()['raw']; setword(raw, core, word, value)
    with pytest.raises(ValueError, match='Empty|batches'):
        timer.parse_records(raw)


@pytest.mark.parametrize('core', [8, 15])
def test_optional_count_intervals_follow_count_ownership_not_packet_tasks(core):
    raw = sample()['raw']
    for word in range(24, 30):
        omitword(raw, core, word)
    records = timer.parse_records(raw)['records']
    assert not any(r['core'] == core and r['stage'] in ('count_ready_poll','cumsum_join','prefix_ready')
                   for r in records)


@pytest.mark.parametrize('word', range(24, 30))
def test_optional_count_interval_cannot_lose_one_endpoint(word):
    raw = sample()['raw']; omitword(raw, 8, word)
    with pytest.raises(ValueError, match='Incomplete optional'):
        timer.parse_records(raw)


def test_analysis_local_scope_round_stats_and_missing_data(tmp_path):
    paths = inputs(tmp_path/'ll_reference_profile', ranks=range(16,32), world=32)
    report = analyzer.analyze(paths)
    assert report['analysis_scope'] == 'local_rank_subset'
    assert report['analyzed_ranks'] == list(range(16,32))
    assert report['rank_kernel'][0]['mean'] == pytest.approx(.27)
    assert report['rank_kernel'][0]['p50'] == pytest.approx(.27)
    with pytest.raises(ValueError, match='Missing ranks'): analyzer.analyze(paths, require_all_ranks=True)
    with pytest.raises(ValueError, match='Missing ranks'): analyzer.analyze(paths[2:])
    with pytest.raises(ValueError, match='misaligned'): analyzer.analyze(paths[:-1])
    with pytest.raises(ValueError, match='Duplicate'): analyzer.analyze(paths+[paths[0]])


@pytest.mark.parametrize('field', ['source', 'config', 'route', 'comparison', 'sampling', 'seed'])
def test_analysis_rejects_mixed_identity(tmp_path, field):
    paths = inputs(tmp_path/'ll_reference_profile')
    saved = json.loads(paths[-1].read_text())
    saved['metadata'][field] = 123 if field == 'seed' else {'changed': True}
    paths[-1].write_text(json.dumps(saved))
    with pytest.raises(ValueError): analyzer.analyze(paths)


def test_discovery_keeps_rank_directories_in_one_case(tmp_path):
    left = tmp_path/'case_0/normal/ll_reference_profile'
    right = tmp_path/'case_1/hotspot/ll_reference_profile'
    inputs(left); inputs(right)
    assert analyzer.discover(tmp_path) == [left, right]
