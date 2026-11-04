"""CPU-only timer ABI, corruption, retry-free capture and aggregation tests."""
import copy
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
timer=load('stage_timer',ROOT/'deep_ep/utils/ll_stage_timer.py')
analyzer=load('stage_analyzer',Path(__file__).with_name('analyze.py'))


def raw(cores=16):
    result=[[[0]*8 for _ in range(timer.SLOTS)] for _ in range(cores)]
    for core,rows in enumerate(result):
        rows[0]=[timer.MAGIC,1,1,cores,core,1,1,len(timer.LEGACY_STAGES)]
        rows[1]=[100,200,100,1,100,0,0,0]
        rows[15]=[110,180,30,3,170,0,0,0]
    return result


def test_layout_names_and_units():
    header=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_stage_timer_layout.h').read_text(encoding='utf-8')
    names=re.search(r'kNames\[\] = \{(.*?)\};',header,re.S).group(1)
    assert re.findall(r'"([a-z_]+)"',names)==list(timer.STAGES)
    report=timer.parse_records(raw(),'dispatch',50_000_000)
    wait=[r for r in report['records'] if r['stage']=='recv_wait'][0]
    assert wait['total_ticks']==30 and wait['total_us']==pytest.approx(.6)
    assert wait['calls']==3 and wait['last_tick']-wait['first_tick']==70
    assert len(report['records'])==32


@pytest.mark.parametrize('corruption',['header','unfinished','active','order','unknown','zero_count'])
def test_reject_corrupted_traces(corruption):
    values=raw()
    if corruption=='header':values[2][0][4]=7
    if corruption=='unfinished':values[2][0][6]=0
    if corruption=='active':values[2][15][5]=1
    if corruption=='order':values[2][15][2]=200
    if corruption=='unknown':values[2][31][0]=1
    if corruption=='zero_count':values[2][15][3]=0
    with pytest.raises(ValueError):timer.parse_records(values,'dispatch')


def test_aggregator_rank_audit_and_percentiles(tmp_path):
    paths=[]
    for rank in range(2):
        for iteration in range(3):
            p=tmp_path/f'{rank}-{iteration}.json'
            p.write_text(json.dumps(dict(raw=raw(),operation='dispatch',frequency_hz=None,
                metadata=dict(rank=rank,world_size=2,iteration=iteration,config={'B':32},correctness='passed'))))
            paths.append(p)
    report=analyzer.analyze(paths)
    assert len(report['per_rank_core_stage'])==64
    assert report['per_rank_core_stage'][0]['samples']==3
    assert 'p90' in report['per_rank_core_stage'][0]
    with pytest.raises(ValueError,match='Missing ranks'):analyzer.analyze(paths[:3])
    with pytest.raises(ValueError,match='misaligned'):analyzer.analyze(paths[:-1])
    with pytest.raises(ValueError,match='Duplicate'):analyzer.analyze(paths+[paths[0]])


def test_context_is_thread_local_and_rejects_extra_calls(monkeypatch):
    class FakeTrace:
        def __init__(self,buffer,operation,mode,frequency):
            self.operation=operation;self.mode=mode;self.launches=0
            self.tensor=SimpleNamespace(device='npu:0')
        def complete(self):assert self.launches==1
    monkeypatch.setattr(timer,'StageTrace',FakeTrace)
    buffer=object();tensors={'input':SimpleNamespace(device='npu:0')}
    with timer.capture(buffer,'dispatch'):
        timer.attach(buffer,tensors,'dispatch')
        assert tensors['ll_stage_timer_mode']==1
        with pytest.raises(RuntimeError,match='multiple'):timer.attach(buffer,tensors,'dispatch')
        with pytest.raises(RuntimeError,match='Nested'):
            with timer.capture(buffer,'dispatch'):pass
    assert not timer._state.active
    ordinary={'input':SimpleNamespace(device='npu:0')}
    timer.attach(buffer,ordinary,'dispatch')
    assert list(ordinary)==['input']


@pytest.mark.parametrize('with_tail', [False, True])
def test_native_st_reference_schedule(monkeypatch,tmp_path,with_tail):
    import sys
    st=load('ll_stage_st',ROOT/'tests/elastic/test_ep_low_latency.py')
    calls=[]
    class Tensor:
        def fill_(self,value):calls.append('fill')
    class Profiler:
        def __enter__(self):return self
        def __exit__(self,*args):calls.append('profile_closed')
        def step(self):calls.append('step')
    def profile(**kwargs):
        assert kwargs['schedule']==dict(wait=0,warmup=1,active=1,repeat=1,skip_first=0)
        return Profiler()
    def gather(out,values):out[:]=[values]
    dist=SimpleNamespace(all_reduce=lambda *a,**k:calls.append('reduce'),barrier=lambda:calls.append('barrier'),
                         ReduceOp=SimpleNamespace(SUM=0),get_rank=lambda:0,get_world_size=lambda:1,all_gather_object=gather)
    torch=SimpleNamespace(float32=0,zeros=lambda *a,**k:Tensor(),ones=lambda *a,**k:Tensor(),
                          npu=SimpleNamespace(synchronize=lambda:calls.append('sync')),distributed=dist,
                          exp=lambda *a,**k:calls.append('exp'))
    npu=SimpleNamespace(profiler=SimpleNamespace(profile=profile,schedule=lambda **kw:kw,
        tensorboard_trace_handler=lambda path:None,_ExperimentalConfig=lambda **kw:kw,
        AiCMetrics=SimpleNamespace(PipeUtilization=0),ProfilerLevel=SimpleNamespace(Level1=0),
        ExportType=SimpleNamespace(Text=0)))
    for name,value in [('torch',torch),('torch.distributed',dist),('torch_npu',npu)]:monkeypatch.setitem(sys.modules,name,value)
    st.completion_boundary=lambda args:None
    st.agree_check=lambda fn,*args:fn()
    st.trace_call=lambda args,label,fn,*a,**kw:fn(*a,**kw)
    st.kernel_durations=lambda *args:calls.append('parse') or [1.0]*100
    args=st.parse_args([])
    assert (args.warmup,args.num_iters,args.perf_tail)==(10,100,50)
    tail=(lambda delay,alignment:calls.append('tail')) if with_tail else None
    st.measure(lambda:calls.append('invoke'),args,'dispatch',tmp_path,after_profile=tail)
    assert calls==['reduce']+['invoke']*10+['sync']+['exp']*200+['step']+['invoke']*100+['barrier','sync','step','profile_closed']+(['tail'] if with_tail else [])+['parse']


def test_moved_checkout_with_spaces_and_other_cwd(tmp_path):
    import os, shutil, subprocess, sys
    moved=tmp_path/'renamed checkout with spaces'
    for relative in ('deep_ep/utils/ll_stage_timer.py','deep_ep/utils/ll_expert_timer.py','tests/ll_stage_timer/analyze.py',
                     'tests/elastic/test_ep_low_latency.py','scripts/ll_timer_env.sh',
                     'tests/ll_rdma/report_tables.py','tests/ll_reference_profile/export_xlsx.py'):
        target=moved/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    away=tmp_path/'different cwd';away.mkdir()
    script=moved/'tests/elastic/test_ep_low_latency.py'
    code='import runpy,sys; m=runpy.run_path(sys.argv[1]); print(m["parse_args"](["--output-dir","results"]).output_dir)'
    output=subprocess.check_output([sys.executable,'-c',code,str(script)],cwd=away,text=True).strip()
    assert Path(output)==moved/'results'
    for rank in range(2):
        path=moved/f'data/rank_{rank}/sample_0000.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(raw=raw(),operation='dispatch',frequency_hz=None,
            metadata=dict(rank=rank,world_size=2,iteration=0,config={'B':32},correctness='passed'))))
    subprocess.run([sys.executable,str(moved/'tests/ll_stage_timer/analyze.py'),'data','--output-dir','report'],cwd=away,check=True)
    assert (moved/'report/stages.csv').is_file()
    env=dict(os.environ,ASCEND_HOME_PATH=str(tmp_path/'synthetic-toolchain'))
    env.pop('CANN_ENV',None)
    path_command = 'cygpath -aw "$LL_TIMER_ROOT"' if os.name == 'nt' else 'printf "%s" "$LL_TIMER_ROOT"'
    actual=subprocess.check_output(['bash','-c','source "$1"; '+path_command,'bash',
                                    str(moved/'scripts/ll_timer_env.sh')],cwd=away,env=env,text=True).strip()
    assert Path(actual)==moved
