"""Configuration guards, shared launch plan, group selection and comparison semantics."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
st=load('parity_st','tests/elastic/test_ep_low_latency.py')
comparison=load('parity_compare','tests/ll_stage_timer/compare.py')


def identity():
    args=st.parse_args(['--num-processes','16','--num-tokens','32','--num-max-tokens-per-rank','256',
                       '--use-1g-huge-pages','0','--ll-stage-timer'])
    args.runtime_identity=dict(implementation='configurable-groups',transport='mte',active_aiv=56,
        logical_group=4,core_protocol=5,stage_timer=True,stage_timer_schema=2,rdma_compiled=False)
    return st.comparison_metadata(args,16,256)


def reports():
    left=dict(comparison=identity(),per_rank_core_stage=[dict(operation='dispatch',rank=0,core=28,stage='kernel',unit='us',mean=10)])
    right=copy.deepcopy(left);right['comparison'].update(implementation='ndr-hns1825',transport='rdma',rdma_compiled=True)
    right['per_rank_core_stage'][0]['mean']=8
    return left,right


def test_attribute_and_function_group_queries():
    assert st.compiled_group(SimpleNamespace(low_latency_ranks_per_group=4),16)==4
    assert st.compiled_group(SimpleNamespace(low_latency_ranks_per_group=lambda p:4 if p<=16 else 16),32)==16
    for bad in (0,17,True,3):
        with pytest.raises(RuntimeError):st.compiled_group(SimpleNamespace(low_latency_ranks_per_group=bad),16)


@pytest.mark.parametrize('key,value',[('active_aiv',64),('seed',2),('device_busy_cycles',40_000_000),
                                    ('timer_mode','drain'),('stage_timer',False),('perf_enabled',False),('num_iters',50)])
def test_mismatched_experiments_cannot_report_speedup(key,value):
    left,right=reports();right['comparison'][key]=value
    with pytest.raises(ValueError,match='Incomparable'):comparison.compare(left,right,'stage')


def test_old_per_iteration_reduce_cannot_be_compared_with_sleep_batch():
    left, right = reports()
    left['comparison']['sampling_protocol'] = 'exp-reduce-stable-tail-v2'
    assert right['comparison']['sampling_protocol'] == 'initial-reduce-sleep-batch-v4'
    assert left['comparison']['device_busy_cycles'] == right['comparison']['device_busy_cycles']
    with pytest.raises(ValueError, match='sampling_protocol'):
        comparison.compare(left, right, 'stage')


def test_old_random_normal_cannot_be_compared_with_balanced_normal():
    left, right = reports()
    left['comparison'].pop('routing')
    with pytest.raises(ValueError, match='routing'):
        comparison.compare(left, right, 'stage')


def test_only_matching_stages_receive_ratios():
    left,right=reports()
    right['per_rank_core_stage'].append(dict(operation='dispatch',rank=0,core=28,stage='send_rdma_body_wait',unit='us',mean=2))
    rows=comparison.compare(left,right,'stage')
    assert rows[0]['candidate_over_baseline']==pytest.approx(.8)
    assert rows[1]['status']=='missing_on_one_side' and rows[1]['candidate_over_baseline'] is None
    del left['comparison']
    with pytest.raises(ValueError,match='Missing comparison'):comparison.compare(left,right,'stage')


def test_different_group_sizes_are_allowed_and_reported():
    left,right=reports();left['comparison']['logical_group']=16
    result=comparison.compare(left,right,'stage')[0]
    assert (result['baseline_group'],result['candidate_group'])==(16,4)
    assert result['candidate_over_baseline']==pytest.approx(.8)


def test_perf_and_expert_comparison_keep_group_as_variable():
    base=identity();base['logical_group']=16
    candidate=copy.deepcopy(base);candidate.update(logical_group=4,implementation='ndr-hns1825',transport='rdma')
    def perf(config,value):
        return dict(status='completed',results=[dict(case_id=0,correctness='passed',comparison=config,
                    dispatch={'ranks':[{'rank':0,'mean_us':value}]})])
    result=comparison.compare(perf(base,10),perf(candidate,8),'perf')[0]
    assert result['candidate_over_baseline']==pytest.approx(.8)
    def experts(config):
        return dict(comparison=config,per_expert=[dict(executing_rank=0,expert_id=0,expert_rank=0,
                    stage='packets',unit='us',mean=None)])
    assert comparison.compare(experts(base),experts(candidate),'expert')[0]['candidate_over_baseline'] is None


@pytest.mark.parametrize('implementation,rdma',[('configurable-groups',False),('ndr-hns1825',True)])
def test_portable_launcher_plan_and_backend_guard(tmp_path,implementation,rdma):
    moved=tmp_path/'checkout with spaces'
    for path in ('tests/ll_stage_timer/run.py','tests/elastic/test_ep_low_latency.py'):
        target=moved/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
    (moved/'tests/ll_stage_timer/branch.json').write_text(json.dumps(dict(implementation=implementation,rdma=rdma)))
    env=dict(os.environ)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE'):env.pop(key,None)
    command=[sys.executable,str(moved/'tests/ll_stage_timer/run.py'),'--backend','mte','--validation','--plan',
             '--num-processes','16']
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['world_size']==16
    command[command.index('mte')]='rdma'
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert (result.returncode==0)==rdma
    if not rdma:assert 'MTE-only' in result.stderr
    # No backend/process-count options: branch default and explicit device list
    # are sufficient; A5 --plan models 16 ranks without launching devices.
    default=[sys.executable,str(moved/'tests/ll_stage_timer/run.py'),'--validation','--plan']
    env['DEEPEP_LL_BACKEND']='rdma' if not rdma else 'mte'  # stale opposite-branch shell state
    result=subprocess.run(default,cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['world_size']==16



    config=moved/'settings.json'
    config.write_text(json.dumps({'backend':'rdma','num_processes':16}))
    configured=[sys.executable,str(moved/'tests/ll_stage_timer/run.py'),'--validation','--plan','--config','settings.json']
    result=subprocess.run(configured,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert (result.returncode==0)==rdma
    overridden=configured+['--backend','mte']
    subprocess.run(overridden,cwd=tmp_path,env=env,capture_output=True,text=True,check=True)


def test_schema_difference_only_allowed_with_both_timers_compiled_off():
    left, right = identity(), identity()
    left['stage_timer_schema'], right['stage_timer_schema'] = 2, 3
    for record in (left, right):
        record.update(stage_timer=False, stage_capture=False, expert_capture=False)
    comparison.check_identity(left, right)
    for record in (left, right):
        record['stage_timer'] = True
    with pytest.raises(ValueError, match='stage_timer_schema'):
        comparison.check_identity(left, right)
