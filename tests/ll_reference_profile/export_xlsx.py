"""Portable, streaming OOXML writer for validated reference-cycle reports.

Only Python's standard library is required on the target machine. Original
uint64 values with more than 15 decimal digits are written as text; subtraction
has already occurred in Python's integer domain before Excel sees a duration.
"""
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
SHEET_NAMES = ('采样汇总', '逐核耗时', '阶段说明', '原始Cycle', 'S包时序', 'R包顺序', '包记录覆盖', '包原始记录')
IDENTITY_HEADERS = ('Rank', '核号', '收发角色', '任务数')
EXACT_LIMIT = 999_999_999_999_999


def column(index):
    name = ''
    while index:
        index, digit = divmod(index - 1, 26)
        name = chr(65 + digit) + name
    return name


def _text(value):
    value = str(value)
    if any(ord(c) < 32 and c not in '\n\r\t' for c in value):
        raise ValueError('Unsupported XML control character in report text')
    if len(value) > 32767:
        raise ValueError('Excel cell exceeds 32767 characters')
    return escape(value)


def _cell(address, value, style=0):
    if value is None:
        return f'<c r="{address}" s="{style}"/>'
    if isinstance(value, dict):
        formula, cached = value['formula'], value['value']
        return (f'<c r="{address}" s="{style}"><f>{_text(formula)}</f>'
                f'<v>{_number(cached)}</v></c>')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, int) and abs(value) > EXACT_LIMIT:
            value = str(value)
            style = 6  # Excel text format, in addition to inlineStr storage.
        else:
            return f'<c r="{address}" s="{style}"><v>{_number(value)}</v></c>'
    return (f'<c r="{address}" s="{style}" t="inlineStr"><is>'
            f'<t xml:space="preserve">{_text(value)}</t></is></c>')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Non-finite or nonnumeric spreadsheet result')
    return str(value)


def _sheet(archive, number, rows, count, columns, widths, header=1, freeze=0, numeric_from=None, explanatory=None):
    if explanatory is None:
        explanatory = number in (1, 3)
    if count > 1048576 or columns > 16384:
        raise ValueError('Excel worksheet capacity exceeded')
    with archive.open(f'xl/worksheets/sheet{number}.xml', 'w') as stream:
        def write(value):
            stream.write(value.encode('utf-8'))
        write(f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{NS}">')
        write(f'<dimension ref="A1:{column(columns)}{count}"/><sheetViews><sheetView showGridLines="0" workbookViewId="0">')
        if freeze:
            write(f'<pane xSplit="{freeze}" ySplit="{header}" topLeftCell="{column(freeze+1)}{header+1}" activePane="bottomRight" state="frozen"/>')
        write('</sheetView></sheetViews><sheetFormatPr defaultRowHeight="22"/><cols>')
        for c in range(1, columns + 1):
            width = widths[c-1] if c <= len(widths) else widths[-1]
            write(f'<col min="{c}" max="{c}" width="{width}" customWidth="1"/>')
        write('</cols><sheetData>')
        actual = 0
        for r, values in enumerate(rows, 1):
            actual = r
            height = 60 if r == header else (44 if explanatory else 22)
            wrapped = list(enumerate(values)) if explanatory else list(enumerate(values[:1]))
            if r != header:
                for c, value in wrapped:
                    if isinstance(value, str):
                        width = widths[c] if c < len(widths) else widths[-1]
                        chars = sum(2 if ord(ch) > 255 else 1 for ch in value)
                        height = max(height, 15 * math.ceil(chars / max(1, width - 2)))
            write(f'<row r="{r}" ht="{height}" customHeight="1">')
            for c, value in enumerate(values, 1):
                if r == header:
                    style = 1
                else:
                    stripe = not explanatory and r % 2 == 0
                    numeric = numeric_from is not None and c >= numeric_from
                    style = (4 if stripe else 3) if numeric else (2 if stripe else 0)
                    if explanatory:
                        style = 5  # wrapped explanatory cells, no truncation
                    elif c == 1:
                        style = 7 if stripe else 5
                write(_cell(f'{column(c)}{r}', value, style))
            write('</row>')
        if actual != count:
            raise ValueError('Spreadsheet row count mismatch')
        write('</sheetData>')
        if not explanatory:
            write(f'<autoFilter ref="A{header}:{column(columns)}{count}"/>')
        write('</worksheet>')


def _package(archive, names=SHEET_NAMES):
    if not names or len(set(names)) != len(names) or any(not n or len(n)>31 or any(c in n for c in '[]:*?/\\') for n in names):
        raise ValueError('Invalid or duplicate worksheet names')
    numbers = range(1, len(names)+1)
    types = ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in numbers)
    archive.writestr('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'+types+'</Types>')
    archive.writestr('_rels/.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="'+REL+'/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    sheets = ''.join(f'<sheet name="{escape(name, {chr(34): "&quot;"})}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(names, 1))
    archive.writestr('xl/workbook.xml', f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{NS}" xmlns:r="{REL}"><bookViews><workbookView/></bookViews><sheets>{sheets}</sheets><calcPr calcMode="auto" fullCalcOnLoad="1"/></workbook>')
    relations = ''.join(f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in numbers)
    archive.writestr('xl/_rels/workbook.xml.rels', f'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{relations}<Relationship Id="rId{len(names)+1}" Type="{REL}/styles" Target="styles.xml"/></Relationships>')
    archive.writestr('xl/styles.xml', f'''<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{NS}">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts>
<fonts count="2"><font><sz val="10"/><name val="Arial"/><color rgb="FF172B3A"/></font><font><b/><sz val="10"/><name val="Arial"/><color rgb="FFFFFFFF"/></font></fonts>
<fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF173D62"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFE6F1F8"/></patternFill></fill></fills>
<borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="8"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="164" fontId="0" fillId="3" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center" wrapText="1"/></xf><xf numFmtId="49" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0"><alignment vertical="center" wrapText="1"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>''')


def validate_selected(report, selected_saved, timer):
    """Do not allow a mismatched single-iteration view beside a valid summary."""
    selected, seen, iterations = [], set(), set()
    expected = set(report['analyzed_ranks'])
    for saved in selected_saved:
        meta = saved['metadata']
        parsed = timer.parse_records(saved['raw'], saved['operation'], report['frequency_hz'])
        rank = meta['rank']
        if rank in seen:
            raise ValueError('Duplicate selected rank')
        seen.add(rank); iterations.add(meta['iteration'])
        if saved.get('format') != timer.FORMAT or meta.get('correctness') != 'passed':
            raise ValueError('Selected data lacks reference format/correctness')
        for key in ('world_size', 'run_id', 'case_id', 'seed', 'config', 'sampling', 'route', 'comparison', 'source', 'profile_seed', 'route_sha256'):
            if meta.get(key) != report.get(key):
                raise ValueError('Selected metadata mismatch: ' + key)
        if (parsed['rank'] != rank or parsed['world_size'] != report['world_size'] or
                parsed['active_aiv'] != report['active_aiv'] or parsed['num_tokens'] != report['num_tokens']):
            raise ValueError('Selected raw identity mismatch')
        selected.append((saved, parsed))
    if seen != expected or len(iterations) != 1 or not iterations.issubset(set(report['iterations'])):
        raise ValueError('Missing ranks or mixed selected iterations')
    return sorted(selected, key=lambda value: value[1]['rank'])


def packet_tables(selected_saved, timer, frequency):
    unit = 'µs' if frequency is not None else 'ticks'
    metrics = timer.SENDER_PACKET_METRICS
    sender = [['Rank','核号','跨机包数','本组包数','lane数','第2跨机包有效','第2包token','第2包header_k',
               '第2包专家','第2包peer','第2包触发整批quiet','跨机与本组时间区间重叠','packet类别顺序交错','quiet批次数','延后提交跨机包数','第二包延后提交',
               *[f'{label} ({unit})' for label in metrics.values()]]]
    receiver = [['Rank','核号','稳定任务j','输出行','发现就绪顺序','就绪批次','处理顺序','源rank','源token',
                 'header_k（Terminal为首跳k）','角色','转发成员数','目标专家','q','槽位']]
    coverage = [['Rank','核号','任务数','记录数','是否截断','lane数','导出耗时 ('+unit+')','DIRECT数（已记录）','FIRST_HOP数（已记录）','Terminal数（已记录）','转发成员数（已记录）']]
    raw = [['Rank','核号',*[f'w{i} {timer.WORDS[i]}' for i in range(64,128)]]]
    for saved in sorted(selected_saved, key=lambda value: value['metadata']['rank']):
        parsed = timer.parse_records(saved['raw'], frequency_hz=frequency)
        rank=parsed['rank']
        for core in parsed['cores']:
            d=core['packet_detail']; c=core['core']
            raw.append([rank,c,*d['raw_metadata']])
            if d['kind']=='sender':
                second=d['second'] or {}
                sender.append([rank,c,d['remote_packets'],d['group_packets'],d['lanes'],d['second'] is not None,
                               *[second.get(k) for k in ('token','header_k','expert','peer','triggered_flush')],
                               d['intervals_interleave'],d['packet_order_interleaves'],d['batch_flushes'],d['deferred_packets'],second.get('deferred'),*[d['durations'].get(k) for k in metrics]])
            else:
                coverage.append([rank,c,d['total_tasks'],d['recorded_tasks'],d['truncated'],d['lanes'],
                                 d['export_ticks']*1e6/frequency if frequency is not None else d['export_ticks'],
                                 *[sum(e['packet_role']==role for e in d['entries']) for role in ('DIRECT','FIRST_HOP','TERMINAL')],
                                 sum(e['forward_members'] for e in d['entries'])])
                for e in sorted(d['entries'],key=lambda e:e['process_order']):
                    receiver.append([rank,c,*[e[k] for k in ('task','output_row','ready_order','ready_batch','process_order',
                        'source_rank','source_token','header_k','packet_role','forward_members','destination_expert','q','slot')]])
    return sender,receiver,coverage,raw


def receiver_batch_rows(selected_saved, timer, frequency):
    rows=[['Rank','核号','成功就绪扫描批次','本次就绪包数','已记录包数','本核记录完整',
           '包类型（发现顺序）','源rank（发现顺序）','源token（发现顺序）',
           '包信息（类型,源rank,源token）','稳定任务j','处理序号','转发成员数']]
    def tup(values):
        return '(' + ', '.join(str(v) for v in values) + ')'
    for saved in sorted(selected_saved,key=lambda value:value['metadata']['rank']):
        parsed=timer.parse_records(saved['raw'],frequency_hz=frequency)
        for core in parsed['cores']:
            detail=core['packet_detail']
            if detail['kind']!='receiver': continue
            for batch in range(1,core['batch_count']+1):
                packets=sorted((e for e in detail['entries'] if e['ready_batch']==batch),key=lambda e:e['ready_order'])
                rows.append([parsed['rank'],core['core'],batch,
                    None if detail['truncated'] else len(packets),len(packets),not detail['truncated'],
                    *[tup(e[key] for e in packets) for key in ('packet_role','source_rank','source_token')],
                    tup(tup((e['packet_role'],e['source_rank'],e['source_token'])) for e in packets),
                    *[tup(e[key] for e in packets) for key in ('task','process_order','forward_members')]])
    return rows


def export_workbook(output_path, report, selected_saved, timer):
    selected = validate_selected(report, selected_saved, timer)
    frequency = report['frequency_hz']
    unit = 'µs' if frequency is not None else 'ticks'
    divisor = frequency / 1e6 if frequency is not None else 1
    group = f"{report['run_id']} / {report['case_id']} / {report.get('route') or ''}"
    iteration = selected[0][0]['metadata']['iteration']
    row_count = sum(parsed['active_aiv'] for _, parsed in selected) + 1
    stages = timer.STAGES
    summary = [
        ['LL dispatch 逐核阶段耗时'],
        ['运行 / case / 路由', group],
        ['统计范围', report['analysis_scope'], '实际global ranks', ', '.join(map(str, report['analyzed_ranks']))],
        ['计时口径', '每rank每轮最长单核w21-w0；包含打点指令，不含末尾诊断身份发布/cache写回。非OFF profiler性能。'],
        ['计数器频率 (Hz)', frequency, '单位', unit],
        ['逐核表选轮（从0开始）', iteration, '汇总全部轮次', ', '.join(map(str, report['iterations']))],
        [], ['Rank', '有效轮数', f'均值 ({unit})', f'p50 ({unit})', f'最小 ({unit})', f'最大 ({unit})'],
    ]
    for value in report['rank_kernel']:
        summary.append([value[k] for k in ('rank', 'samples', 'mean', 'p50', 'minimum', 'maximum')])
    summary += [[], ['配置项', '值']]
    for key, value in sorted(report['config'].items()):
        summary.append([key, json.dumps(value, ensure_ascii=False)])
    summary += [['采样', json.dumps(report['sampling'], ensure_ascii=False)],
                ['实际核数 / EP', report['active_aiv'], report['world_size']],
                ['身份字段', '逐核页保留global Rank和核号；省略实验组、服务器、本地卡及轮次列，run/case和选轮见本页。'],
                ['阶段语义', '主阶段包含细分，不能相加；空白=未执行或无点，0=确有记录且耗时为0。']]

    def per_core():
        for saved, parsed in selected:
            by_core = {c['core']: {} for c in parsed['cores']}
            for record in parsed['records']:
                by_core[record['core']][record['stage']] = record
            for core in parsed['cores']:
                index = core['core']; rank = parsed['rank']
                identity = [rank, index, '发送' if core['role'] == 'send' else '接收', core['task_count']]
                words = [v for row in saved['raw'][index] for v in row]
                yield identity, words, by_core[index], core

    def detail_rows():
        yield [*IDENTITY_HEADERS, *[f"{timer.STAGE_INFO[name]['label']} ({unit})" for name in stages], 'packet就绪批次数']
        for row, (identity, words, records, core) in enumerate(per_core(), 2):
            values = []
            for index, name in enumerate(stages):
                rec = records.get(name)
                if rec is None:
                    values.append(None); continue
                info = timer.STAGE_INFO[name]
                start, end = info['start'], info['end']
                if start is None:
                    formula = f"'原始Cycle'!{column(end+3)}{row}"
                elif max(words[start], words[end]) <= EXACT_LIMIT:
                    formula = f"('原始Cycle'!{column(end+3)}{row}-'原始Cycle'!{column(start+3)}{row})"
                else:
                    formula = f"'原始Cycle'!{column(67+index)}{row}"
                value = rec['total_ticks'] / divisor if frequency is not None else rec['total_ticks']
                if frequency is not None:
                    formula += "/'阶段说明'!$B$2"
                if frequency is None and rec['total_ticks'] > EXACT_LIMIT:
                    values.append(str(rec['total_ticks']))
                else:
                    values.append(dict(formula=formula, value=value))
            yield [*identity, *values, core['batch_count']]

    definitions = [
        ['阶段换算', '同核整数cycle差 / 每µs的cycle数；w33/w34直接累计'],
        ['每µs的cycle数', divisor if frequency is not None else None, '无频率时全表保留ticks' if frequency is None else 'GetSystemCycle频率'],
        ['包含关系', '提交仅描述提交；已有异步工作在后续drain中完成。细分项不与主项相加。'],
        [], ['阶段', 'word换算（ticks）', '源码边界 / NDR说明', '包含关系'],
    ]
    for name in stages:
        info = timer.STAGE_INFO[name]
        expression = f"w{info['end']}" if info['start'] is None else f"w{info['end']}-w{info['start']}"
        definitions.append([info['label'], expression, info['boundary'], info['relationship']])
    definitions += [
        ['有效性', 'w58 valid_mask', '未执行点mask未置位且word为0；执行时刻或累计为0仍保留有效位。'],
        ['任务 / 次数', '发送任务w43；接收任务w37；packet batch w41', 'w33/w34的calls对应w41批次数，不是packet总数。'],
        ['原始精度', '超过15位整数保存为文本', 'BO列起保留Python整数域作差得到的33项ticks；大时钟公式引用这些差值，避免Excel大数消减。'],
        ['核时钟', '仅同核相减', '不做跨rank绝对时钟对齐；不从嵌套阶段之和推算总时长。'],
        ['NDR边界', '参考阶段命名保留', '同名SendCnt包含当前NDR已有路由交换；payload提交可能包含RDMA post/quiet；不移植参考业务逻辑。'],
        ['接收批次', '标量累计两段', '只对完整ready批次和加载消费区间累计，不对每次空扫描插点。'],
    ]

    def raw_rows():
        yield ['Rank', '核号', *[f'w{i} {name}' for i, name in enumerate(timer.WORDS[:64])],
               *[f"{timer.STAGE_INFO[name]['label']} 精确ticks（整数作差）" for name in stages]]
        for identity, words, records, _ in per_core():
            yield [identity[0], identity[1], *words[:64],
                   *[records[name]['total_ticks'] if name in records else None for name in stages]]

    packet_sender, _, packet_coverage, packet_raw = packet_tables(selected_saved, timer, frequency)
    packet_receiver = receiver_batch_rows(selected_saved, timer, frequency)
    definitions += [
        ['第2跨机包', '每个S核实际第2次跨机packet', '主q在SendPacket入口之前；调度另扣除成员q和前包提交区间。暂存至put前经过包含下一包准备，剩余等待另列。'],
        ['提交与退休', 'put返回不是远端消费完成', '第二包发送不含自动Flush quiet；批次quiet单列。跨机退休区间结束于末批quiet。'],
        ['交错区间', '同核首开始至末提交', '跨机与本组可能交错；整体区间可以包含另一类工作，不能相加或解释为纯链路传输时间。'],
        ['R包顺序', '记录在UB，w21后导出', '发现顺序不是实际到达顺序；处理顺序按ConsumeLoaded入口。Terminal header_k为首跳返回k。'],
        ['覆盖', '最多128个稳定任务，且仅用剩余UB', '不减少原lane；空间不足截断，包记录覆盖页标记。截断不能当作完整角色/转发计数。'],
        ['导出', 'w102→w103', '业务w21之前不做日志DMA；写出前总耗时w57包括此诊断导出。'],
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # x mode is essential: analysis must never replace an existing measurement.
    with ZipFile(path, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        _package(archive)
        _sheet(archive, 1, summary, len(summary), 6, [35, 68, 24, 48, 24, 24], header=8)
        _sheet(archive, 2, detail_rows(), row_count, 34, [10, 10, 12, 12, 22], freeze=5, numeric_from=5)
        _sheet(archive, 3, definitions, len(definitions), 4, [36, 48, 95, 38], header=5)
        _sheet(archive, 4, raw_rows(), row_count, 66+len(stages), [10, 10, 27], freeze=2)
        for number, rows in enumerate((packet_sender,packet_receiver,packet_coverage,packet_raw),5):
            _sheet(archive, number, rows, len(rows), len(rows[0]), [10,10,24], freeze=2)
    return path
