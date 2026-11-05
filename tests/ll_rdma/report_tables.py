"""Offline wide tables and run-level XLSX assembly; Python standard library only."""
import csv
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ll_table_writer', ROOT/'tests/ll_reference_profile/export_xlsx.py')
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)


def csv_table(path, headers, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        output = csv.writer(stream)
        output.writerow(headers)
        output.writerows(rows)


# 非阶段类原始计数，追加在阶段列之后，便于把阶段耗时与同一核的任务量对照。
REFERENCE_COUNTER_WORDS = (37, 41, 43, 48, 49)


def wide_records(selected, timer, frequency):
    """One row for each operation/rank/core, from exactly one capture iteration."""
    stages = tuple(s for s in timer.STAGES if s != 'header')
    # 旧 stage timer 没有 reference 的 word 名与 valid_mask，仅输出占位列。
    names = getattr(timer, 'WORDS', None)
    counters = tuple(f'w{w} {names[w]}' if names else f'w{w}' for w in REFERENCE_COUNTER_WORDS)
    headers = ['operation', 'rank', 'host_group', 'local_rank', 'core', 'role', 'iteration', 'unit', *stages, *counters]
    rows, calls, raw_rows = [], [], []
    for saved in sorted(selected, key=lambda v: (v['operation'], v['metadata']['rank'])):
        parsed = timer.parse_records(saved['raw'], saved['operation'], frequency)
        rank, iteration = saved['metadata']['rank'], saved['metadata']['iteration']
        by_core = {core: {} for core in range(parsed['active_aiv'])}
        for record in parsed['records']:
            by_core[record['core']][record['stage']] = record
        for core, records in by_core.items():
            identity = [saved['operation'], rank, rank//16, rank%16, core,
                        'send' if core < parsed['active_aiv']//2 else 'receive', iteration,
                        'us' if frequency is not None else 'ticks']
            words = [v for block in saved['raw'][core] for v in block]
            valid = words[58] if names and len(words) > 58 else 0
            # 未置有效位的计数按缺失输出，不能把“未执行”写成 0。
            observed = [words[w] if valid >> w & 1 and w < len(words) else None
                        for w in REFERENCE_COUNTER_WORDS]
            rows.append(identity + [records[s].get('total_us', records[s]['total_ticks']) if s in records else None
                                    for s in stages] + observed)
            calls.append(identity[:7] + ['calls'] + [records[s]['calls'] if s in records else None
                                                    for s in stages] + [None] * len(counters))
            raw_rows.append(identity[:7] + words)
    keys = [tuple(r[i] for i in (0, 1, 4)) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate operation/rank/core in selected wide table')
    return stages, headers, rows, calls, raw_rows


def mean_rows(report, headers, stages):
    groups = {}
    for r in report['per_rank_core_stage']:
        key = (r.get('operation', 'dispatch'), r['rank'], r['core'], r['role'])
        groups.setdefault(key, {})[r['stage']] = r
    # 均值页没有逐轮原始计数，仅在尾部补齐占位列，保持与宽表同一列数。
    padding = len(headers) - 8 - len(stages)
    result = []
    for (operation, rank, core, role), records in sorted(groups.items()):
        result.append([operation, rank, rank//16, rank%16, core, role, 'all_mean',
                       'us' if report['frequency_hz'] is not None else 'ticks'] +
                      [records[s]['mean'] if s in records else None for s in stages] + [None] * padding)
    return result


def native_workbook(output, report, selected, timer):
    """The legacy schema keeps its actual phases; never impersonate reference boundaries."""
    output = Path(output)
    stages, headers, rows, calls, raw = wide_records(selected, timer, report['frequency_hz'])
    means = mean_rows(report, headers, stages)
    csv_table(output/'stages.csv', headers, rows)
    csv_table(output/'stages_mean.csv', headers, means)
    csv_table(output/'stage_calls.csv', headers, calls)
    operations = sorted({r[0] for r in rows})
    frequency = report['frequency_hz']
    unit = 'µs' if frequency is not None else 'ticks'
    chinese = ['算子', 'Rank', '服务器组', '本地卡', '核号', '收发分工', '采集轮次', '单位']
    summary = [['LL阶段采样汇总', '值'], ['run / case', f"{report.get('run_id')} / {report.get('case_id')}"],
               ['分析范围', report['analysis_scope']], ['实际global ranks', str(report['analyzed_ranks'])],
               ['逐核页', f"选中第{rows[0][6]}轮（从0开始），每个算子/rank/core仅一行"],
               ['均值页', '全部capture中各阶段有记录轮次的均值；未执行不补0'],
               ['计时口径', '旧stage timer累计区间；父子阶段不能相加，observe异步提交不等于完成'],
               ['性能基线', '未提供OFF profiler对照；此处不能伪造性能对照'],
               ['配置', json.dumps(report.get('configurations', {}), ensure_ascii=False)],
               ['采样', json.dumps(report.get('sampling'), ensure_ascii=False)]]
    definitions = [['阶段换算', '累计ticks / 每µs cycles'], ['每µs cycles', frequency/1e6 if frequency else None],
                   ['格式', f"stage schema {report['schema']}；与reference的33阶段不是同一打点布局"],
                   ['缺失与零', '空白=无记录；0=已执行且累计为0。calls页记录选轮调用次数。'],
                   ['阶段标签', '含义']]
    definitions += [[s, '累计区间；可能与父子阶段嵌套，不能求和'] for s in stages]
    definitions += [['核角色', 'send/receive为编号分工；combine分别包含DIRECT回传、partial/聚合等任务'],
                    ['原始精度', '超过15位原始word存文本；耗时取累计整数，不用跨核或跨rank时间戳相减'],
                    ['统计明细', '阶段统计页保留全部轮次mean/p50/p90/p99/min/max及有效samples']]
    specs = [('采样汇总', summary, [30, 100], True, None)]
    for operation in operations:
        suffix = '' if len(operations) == 1 else '_'+operation
        active = [s for s in stages if any(r.get('operation', 'dispatch') == operation and r['stage'] == s
                                         for r in report['per_rank_core_stage'])]
        indices = [stages.index(s) for s in active]
        head = chinese + [f'{s} ({unit})' for s in active]
        specs.append(('逐核耗时'+suffix, [head]+[[*r[:8], *[r[8+i] for i in indices]] for r in rows if r[0] == operation], [16, 10, 12, 10, 10, 12, 12, 10, 24], False, 9))
        specs.append(('逐核均值'+suffix, [head]+[[*r[:8], *[r[8+i] for i in indices]] for r in means if r[0] == operation], [16, 10, 12, 10, 10, 12, 12, 10, 24], False, 9))
    specs += [('阶段说明', definitions, [36, 110], True, None),
              ('原始Cycle', [chinese[:7]+[f'w{i}' for i in range(len(raw[0])-7)]]+raw, [16, 10, 12, 10, 10, 12, 12, 25], False, None),
              ('阶段调用次数', [chinese+list(stages)]+[r[:8+len(stages)] for r in calls], [16, 10, 12, 10, 10, 12, 12, 10, 24], False, None)]
    for filename, title in [('stages_long.csv', '阶段统计'), ('bandwidth.csv', '通信计数'),
                            ('recv_core_samples.csv', '接收明细'), ('experts.csv', '专家统计'), ('expert_cores.csv', '专家逐核')]:
        source = output/filename
        if source.is_file():
            with source.open(encoding='utf-8-sig', newline='') as stream:
                table = list(csv.reader(stream))
            # Keep CSV numeric fields numeric in Excel, without risking uint64 rounding.
            for r in range(1, len(table)):
                table[r] = [number(v) for v in table[r]]
            specs.append((title, table, [24], False, None))
    workbook(output/'core_stage_timings.xlsx', specs)


def number(value):
    if value == '':
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def display_table(rows, widths):
    """Compact Excel view; CSV keeps full identity for offline joins and audit."""
    drop = {'实验组', '服务器', '服务器组', '本地卡', '采集轮次',
            'host_group', 'local_rank', 'iteration'}
    pinned = {'算子', 'operation', 'Rank', 'rank', '核号', 'core',
              '收发分工', '收发角色', 'role', '任务数', 'task_count',
              'executing_rank', 'expert_rank', 'expert_id'}
    head = rows[0]
    keep = [i for i, h in enumerate(head) if h not in drop]
    frozen = [i for i in keep if head[i] in pinned or head[i] == 'kernel' or head[i].startswith('kernel (')]
    order = frozen + [i for i in keep if i not in frozen]
    return ([[r[i] if i < len(r) else None for i in order] for r in rows],
            [widths[min(i, len(widths)-1)] for i in order], len(frozen))


def workbook(path, specs):
    with ZipFile(path, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        writer._package(archive, [s[0] for s in specs])
        for i, (_, rows, widths, explanatory, numeric_from) in enumerate(specs, 1):
            freeze = 0
            if not explanatory:
                old_headers = rows[0]
                numeric_headers = set(old_headers[numeric_from-1:]) if numeric_from else set()
                rows, widths, freeze = display_table(rows, widths)
                numeric_from = next((j+1 for j, h in enumerate(rows[0]) if h in numeric_headers), None)
            writer._sheet(archive, i, rows, len(rows), max(map(len, rows)), widths,
                          freeze=freeze, numeric_from=numeric_from,
                          explanatory=explanatory)


def merge_workbooks(output, entries):
    """Bundle independent cases as separate tabs; retarget formulas and keep cached values."""
    ns = {'m': writer.NS}
    names, parts, extra, index = [], [], [], [['组', 'run / case', '模式', '配置', 'global ranks', '逐核轮次']]
    styles = None
    for group, (path, report, kind) in enumerate(entries, 1):
        prefix = f'C{group:03d}_' if len(entries) > 1 else ''
        with ZipFile(path) as source:
            current_style = source.read('xl/styles.xml')
            if styles is not None and styles != current_style:
                raise ValueError('Workbook styles differ; cannot merge without preserving them')
            styles = current_style
            sheets = ET.fromstring(source.read('xl/workbook.xml')).findall('m:sheets/m:sheet', ns)
            mapping = {s.attrib['name']: prefix+s.attrib['name'] for s in sheets}
            for i, s in enumerate(sheets, 1):
                name = mapping[s.attrib['name']]
                data = source.read(f'xl/worksheets/sheet{i}.xml')
                for old, new in mapping.items():
                    data = data.replace(("'"+old+"'!").encode('utf-8'), ("'"+new+"'!").encode('utf-8'))
                names.append(name); parts.append(data)
        if kind == 'reference':
            for filename, title in [('stages_mean.csv', '逐核均值'), ('stage_calls.csv', '阶段调用次数'),
                                    ('stages_long.csv', '全部轮次明细')]:
                table_path = Path(path).parent/filename
                if table_path.is_file():
                    with table_path.open(encoding='utf-8-sig', newline='') as stream:
                        rows = list(csv.reader(stream))
                    for r in range(1, len(rows)):
                        rows[r] = [number(v) for v in rows[r]]
                    extra.append((prefix+title, rows))
        index.append([prefix or '本次', f"{report.get('run_id')} / {report.get('case_id')}", kind,
                      json.dumps(report.get('config', report.get('configurations')), ensure_ascii=False),
                      str(report.get('analyzed_ranks')), report.get('selected_iteration')])
    if len({report.get('run_id') for _, report, _ in entries}) > 1:
        raise ValueError('Do not combine different run IDs into one run workbook')
    names += [name for name, _ in extra]
    if len(entries) > 1:
        names.insert(0, '报告索引')
    with ZipFile(output, 'x', compression=ZIP_DEFLATED, compresslevel=6) as target:
        writer._package(target, names)
        first = 1
        if len(entries) > 1:
            writer._sheet(target, 1, index, len(index), 6, [12, 40, 14, 80, 40, 12], explanatory=True)
            first = 2
        for i, data in enumerate(parts, first):
            target.writestr(f'xl/worksheets/sheet{i}.xml', data)
        for i, (_, rows) in enumerate(extra, first+len(parts)):
            rows, widths, freeze = display_table(rows, [18, 12, 12, 12, 12, 12, 14, 12, 24])
            writer._sheet(target, i, rows, len(rows), len(rows[0]), widths,
                          freeze=freeze, explanatory=False)
