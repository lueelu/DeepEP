"""Compare matching LL reports; never treat differing configurations as a speedup."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DIFFERENCES={'implementation','transport','rdma_compiled','source_revision','source_dirty','logical_group'}
REQUIRED={'implementation','transport','world_size','logical_group','active_aiv','core_protocol',
          'case','warmup','num_iters','perf_tail','delay_rows','delay_cols','stage_capture','expert_capture',
          'timer_mode','stage_timer','stage_timer_schema','frequency_hz','seed','route','use_1g_huge_pages',
          'comparison_schema','sampling_protocol','perf_enabled'}


def check_identity(left,right):
    if not isinstance(left,dict) or not isinstance(right,dict) or not REQUIRED<=left.keys() or not REQUIRED<=right.keys():
        raise ValueError('Missing comparison identity; recollect with the shared launcher and matching wheels')
    for item in (left,right):
        if item['comparison_schema']!=1 or any(type(item[key]) is not int or item[key]<=0
                                               for key in ('world_size','logical_group','active_aiv','core_protocol')):
            raise ValueError('Unknown or invalid runtime comparison identity')
    differences = set(DIFFERENCES)
    # Schema-only diagnostic additions cannot affect a build with timer compiled OFF.
    if all(item['stage_timer'] is False and item['stage_capture'] is False
           and item['expert_capture'] is False for item in (left, right)):
        differences.add('stage_timer_schema')
    mismatches=[key for key in sorted(left.keys()|right.keys()) if key not in differences and left.get(key)!=right.get(key)]
    if mismatches: raise ValueError('Incomparable settings: '+', '.join(mismatches))


def compare(left,right,kind):
    def indexed(report):
        if kind=='perf':
            result={}
            if report.get('status')!='completed': raise ValueError('Incomplete performance run')
            for case in report['results']:
                if case.get('correctness')!='passed': raise ValueError('Performance case lacks correctness evidence')
                for operation in ('dispatch','combine'):
                    for rank in case.get(operation,{}).get('ranks',[]):
                        key=(case['case_id'],operation,rank['rank'])
                        if key in result: raise ValueError('Duplicate performance key')
                        result[key]=(rank['mean_us'],case.get('comparison'))
            return result
        collection='per_rank_core_stage' if kind=='stage' else 'per_expert'
        keys=('operation','rank','core','stage') if kind=='stage' else ('executing_rank','expert_id','expert_rank','stage')
        result={}
        for row in report.get(collection,[]):
            key=tuple(row[k] for k in keys)
            if key in result: raise ValueError('Duplicate diagnostic key')
            if row['unit']!='us': raise ValueError('Use microseconds for comparison')
            result[key]=(row['mean'],report.get('comparison'))
        return result
    l,r=indexed(left),indexed(right)
    if not l or not r: raise ValueError('No matching measurement data in one report')
    if not l.keys()&r.keys(): raise ValueError('No overlapping measurement keys')
    if kind!='perf': check_identity(left.get('comparison'),right.get('comparison'))
    rows=[]
    for key in sorted(l.keys()|r.keys()):
        if key in l and key in r: check_identity(l[key][1],r[key][1])
        a=l[key][0] if key in l else None;b=r[key][0] if key in r else None
        rows.append(dict(key=json.dumps(key),baseline_mean_us=a,candidate_mean_us=b,
                         baseline_group=l[key][1]['logical_group'] if key in l else None,
                         candidate_group=r[key][1]['logical_group'] if key in r else None,
                         candidate_over_baseline=b/a if a is not None and a>0 and b is not None else None,
                         status='matched' if key in l and key in r else 'missing_on_one_side'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--kind',choices=('perf','stage','expert'),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    resolve=lambda path:path if path.is_absolute() else ROOT/path
    rows=compare(json.loads(resolve(args.baseline).read_text()),json.loads(resolve(args.candidate).read_text()),args.kind)
    output=resolve(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Compared matching settings -> {output}; missing stages remain blank, never zero-filled')


if __name__=='__main__':main()
