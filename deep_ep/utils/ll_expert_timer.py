"""Decode per-expert receive work; cumulative core time is not end-to-end latency."""
import math

MAGIC = 0x4C4C4531
SCHEMA, HEADER_ROWS, FIELDS = 1, 8, 8
STAGES = ('packets', 'metadata', 'output_write', 'forward')


def parse_records(raw, rank, world, experts, frequency_hz=1e9):
    if not (type(rank) is int and type(world) is int and type(experts) is int and
            0 <= rank < world <= experts <= 1024 and experts % world == 0):
        raise ValueError('Invalid expert timer rank/world/expert configuration')
    if frequency_hz is not None and (not math.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError('Invalid counter frequency')
    cores=len(raw)
    if not 16 <= cores <= 72 or cores%2:
        raise ValueError('Invalid expert timer AIV count')
    records, modes = [], set()
    for core, rows in enumerate(raw):
        if len(rows)!=HEADER_ROWS+experts*len(STAGES) or any(len(row)!=FIELDS for row in rows):
            raise ValueError('Expert timer shape mismatch')
        if any(type(v) is not int or not 0 <= v < 1<<63 for row in rows for v in row):
            raise ValueError('Invalid expert timer word')
        header=rows[0]; mode=header[5]
        if header != [MAGIC,SCHEMA,rank,core,experts,mode,1,len(STAGES)] or mode not in (1,2):
            raise ValueError('Incomplete or mismatched expert timer header')
        if any(v for row in rows[1:HEADER_ROWS] for v in row):
            raise ValueError('Nonzero expert header padding')
        modes.add(mode)
        for expert in range(experts):
            for phase, name in enumerate(STAGES):
                first,last,total,count,start,active,r0,r1=rows[HEADER_ROWS+expert*len(STAGES)+phase]
                if active or r0 or r1:
                    raise ValueError('Unclosed expert interval')
                if not count:
                    if any((first,last,total,start)): raise ValueError('Unfinished expert interval')
                    continue
                if core < cores//2: raise ValueError('Expert records on a sender core')
                if last < first or total > last-first:
                    raise ValueError('Invalid expert interval order')
                owner=expert//(experts//world)
                if phase != 3 and owner != rank:
                    raise ValueError('Local expert work attributed to another rank')
                if phase == 3 and owner//16 != rank//16:
                    raise ValueError('FIRST_HOP forwarding crosses physical hosts')
                if phase == 0:
                    if total or start: raise ValueError('Packet counter must not contain a duration')
                elif not first <= start <= last:
                    raise ValueError('Invalid expert interval start')
                row=dict(executing_rank=rank, expert_id=expert, expert_rank=owner, core=core,
                         stage=name, first_tick=first, last_tick=last, total_ticks=total,
                         calls=count, span_ticks=last-first, time_measured=phase!=0)
                if frequency_hz is not None:
                    row.update(total_us=total*1e6/frequency_hz, span_us=(last-first)*1e6/frequency_hz)
                records.append(row)
    if len(modes)!=1: raise ValueError('Mixed expert boundary modes')
    return dict(schema=SCHEMA, rank=rank, world_size=world, experts=experts, active_aiv=cores,
                boundary_mode='observe' if modes=={1} else 'drain', records=records,
                interpretation='Receiver-side code intervals only. Shared FIRST_HOP preparation and batch waiting are unassigned. Cross-core sums are work, not elapsed latency; packet records count deliveries, not unique input tokens.')
