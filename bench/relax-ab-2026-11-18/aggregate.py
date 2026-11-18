#!/usr/bin/env python3
"""Aggregate valid, paired rounds for this campaign."""
import json
import os
import statistics

ROOT = os.path.dirname(os.path.abspath(__file__))
with open(f'{ROOT}/arms.json') as fh:
    ARMS = json.load(fh)['arms']
EXPECTED_SLOTS = {arm['id']: int(arm['expert_cache_slots']) for arm in ARMS}
METRICS = ('engine_cold', 'engine_warm', 'burst', 'prefill', 'wide')


def filename(name, rnd, ext='json'):
    suffix = '' if rnd == 1 else f'_r{rnd}'
    return f'{ROOT}/results/{name}{suffix}.{ext}'


def load(name, rnd):
    path = filename(name, rnd)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rec = json.load(fh)
    if rec.get('timing_valid') is not True:
        return None
    arm_id = name.removeprefix('ctl_').removeprefix('trt_')
    expected_slots = EXPECTED_SLOTS[arm_id]
    if rec.get('expert_cache_slots') != expected_slots or \
            rec.get('expected_expert_cache_slots') != expected_slots:
        return None
    status_path = filename(name, rnd, 'status.json')
    if os.path.exists(status_path):
        with open(status_path) as fh:
            if json.load(fh).get('status') != 'completed':
                return None
    if rec.get('route_expected') and not rec.get('route_engaged'):
        return None
    if rec.get('route_expected') and not (rec.get('route_evidence') or {}).get('valid'):
        return None
    return rec


def metric(rec, name):
    if name == 'engine_cold':
        values = rec.get('engine_decode_512') or []
        return float(values[0]) if values else None
    if name == 'engine_warm':
        values = rec.get('engine_decode_512') or []
        return float(values[1]) if len(values) > 1 else None
    if name == 'burst':
        return (rec.get('burst') or {}).get('aggregate_tok_s')
    if name == 'prefill':
        return (rec.get('prefill') or {}).get('wall_tok_s')
    if name == 'wide':
        return (rec.get('wide') or {}).get('wall_tok_s')
    return None


def sparse_launches(rec):
    by_width = rec.get('wide_launches_by_t') or (rec.get('route_evidence') or {}).get(
        'wide_launches_by_t') or {}
    return {width: {kind: count for kind, count in kinds.items() if count}
            for width, kinds in by_width.items()
            if any(kinds.values())}


def cross_window_summary(sid):
    ctl_records = [rec for rnd in range(1, 4)
                   if (rec := load('ctl_relaxed_x4_dot4', rnd))]
    trt_records = [rec for rnd in range(1, 4) if (rec := load(f'trt_{sid}', rnd))]
    print(f'CROSS-WINDOW {sid}: treatment-only; control was measured in an earlier window '
          f'and is reused from campaign-9 ctl_relaxed_x4_dot4, not rerun; '
          f'valid rounds ctl={len(ctl_records)}/3 '
          f'treatment={len(trt_records)}/3')
    print('  Round spread is shown as each window\'s independent min-max; these are not paired rounds.')
    for met in METRICS:
        ctl_values = [value for rec in ctl_records if (value := metric(rec, met)) is not None]
        trt_values = [value for rec in trt_records if (value := metric(rec, met)) is not None]
        if not ctl_values or not trt_values:
            print(f'  {met}: unavailable (needs valid control and treatment rounds)')
            continue
        ctl_median = statistics.median(ctl_values)
        trt_median = statistics.median(trt_values)
        delta = 100 * (trt_median - ctl_median) / ctl_median
        ctl_spread = f'{min(ctl_values):.1f}-{max(ctl_values):.1f}'
        trt_spread = f'{min(trt_values):.1f}-{max(trt_values):.1f}'
        print(f'  {met}: ctl median {ctl_median:.1f} (spread {ctl_spread}); '
              f'treatment median {trt_median:.1f} (spread {trt_spread}); '
              f'delta {delta:+.1f}%')


def main():
    arms = ARMS
    ids = [arm['id'] for arm in arms]
    paired_ids = [arm['id'] for arm in arms if not arm.get('treatment_only')]
    paired = {}
    for arm in arms:
        if arm.get('treatment_only'):
            continue
        sid = arm['id']
        ctl_name, trt_name = f'ctl_{sid}', f'trt_{sid}'
        common = []
        for rnd in range(1, 4):
            ctl, trt = load(ctl_name, rnd), load(trt_name, rnd)
            if not ctl or not trt:
                continue
            # Route-enabled arms are interpretable only if the corresponding launch engaged X4.
            if (ctl.get('route_expected') and not ctl.get('route_engaged')) or (
                    trt.get('route_expected') and not trt.get('route_engaged')):
                continue
            expected_slots = EXPECTED_SLOTS[sid]
            if ctl.get('expert_cache_slots') != expected_slots or \
                    trt.get('expert_cache_slots') != expected_slots:
                print(f'{sid} r{rnd}: excluding non-{expected_slots}-slot cache state '
                      f'{ctl.get("expert_cache_slots")} / {trt.get("expert_cache_slots")}')
                continue
            common.append((rnd, ctl, trt))
        ctl_rounds = [rnd for rnd in range(1, 4) if load(ctl_name, rnd)]
        trt_rounds = [rnd for rnd in range(1, 4) if load(trt_name, rnd)]
        unmatched = sorted(set(ctl_rounds) ^ set(trt_rounds))
        if unmatched:
            print(f'{sid}: excluding unpaired valid rounds {unmatched}')
        paired[sid] = common
        ctl_paired = [ctl for _, ctl, _ in common]
        trt_paired = [trt for _, _, trt in common]
        assert len(ctl_paired) == len(trt_paired)
        print(f'{sid}: paired valid rounds {len(common)}/3; cache slots expected={EXPECTED_SLOTS[sid]}, '
              f'actual ctl={[x[1].get("expert_cache_slots") for x in common]}, '
              f'treatment={[x[2].get("expert_cache_slots") for x in common]}')
        route_count = sum((x[1].get('x4_captures') or {}).get('n', 0) +
                          (x[2].get('x4_captures') or {}).get('n', 0) for x in common)
        wide_scope_count = sum((x[1].get('route_evidence') or {}).get('x4_wide_scope_note_count', 0) +
                               (x[2].get('route_evidence') or {}).get('x4_wide_scope_note_count', 0)
                               for x in common)
        projection_count = sum((x[1].get('route_evidence') or {}).get('relaxed_projection_pack_count', 0) +
                               (x[2].get('route_evidence') or {}).get('relaxed_projection_pack_count', 0)
                               for x in common)
        print(f'  X4 total captures={route_count}; wide-scope X4 notes={wide_scope_count}; '
              f'relaxed projection-ready lines={projection_count}')
        for rnd, ctl, trt in common:
            ctl_launches, trt_launches = ctl.get('wide_launches') or {}, trt.get('wide_launches') or {}
            launch_delta = {kind: trt_launches.get(kind, 0) - ctl_launches.get(kind, 0)
                            for kind in ('k0', 'k1', 'k2')}
            ctl_lines, trt_lines = ctl.get('wide_request_lines') or [], trt.get('wide_request_lines') or []
            print(f'  r{rnd}: report-wide T9-T64 launches ctl={sparse_launches(ctl)} '
                  f'treatment={sparse_launches(trt)}; summary delta={launch_delta}; '
                  f'wide request-era lines ctl={len(ctl_lines)} {ctl_lines!r}; '
                  f'treatment={len(trt_lines)} {trt_lines!r}')
    print('Report-wide launch totals do not alone prove request-era routing; treatment validity also '
          'requires a matching T>=9/k0-or-k1 X4 note and an accepted wide-request log line.')

    print(f'{"pair":20s} {"metric":12s} {"ctl med":>9s} {"trt med":>9s} {"delta":>9s} round deltas (%)')
    deltas = {}
    for sid in paired_ids:
        deltas[sid] = {}
        for met in METRICS:
            vals = []
            for rnd, ctl, trt in paired[sid]:
                cv, tv = metric(ctl, met), metric(trt, met)
                if cv is not None and tv is not None and cv > 0:
                    vals.append((rnd, cv, tv, 100 * (tv - cv) / cv))
            ctl_vals = [v[1] for v in vals]
            trt_vals = [v[2] for v in vals]
            assert len(ctl_vals) == len(trt_vals)
            if not vals:
                print(f'{sid:20s} {met:12s} unavailable (no valid paired rounds)')
                continue
            ctl_med = statistics.median(v[1] for v in vals)
            trt_med = statistics.median(v[2] for v in vals)
            delta = 100 * (trt_med - ctl_med) / ctl_med
            deltas[sid][met] = [abs(v[3]) for v in vals] if sid == 'aa_floor' else [v[3] for v in vals]
            detail = ', '.join(f'r{rnd} {d:+.1f}' for rnd, _, _, d in vals)
            print(f'{sid:20s} {met:12s} {ctl_med:9.1f} {trt_med:9.1f} {delta:+8.1f}% {detail}')

    if 'aa_floor' in ids:
        print('A/A floor (absolute round deltas):')
        for met in METRICS:
            vals = deltas.get('aa_floor', {}).get(met, [])
            if vals:
                print(f'  {met}: median {statistics.median(vals):.1f}%, max {max(vals):.1f}%')
            else:
                print(f'  {met}: unavailable (no valid paired rounds)')

    for arm in arms:
        if arm.get('treatment_only'):
            cross_window_summary(arm['id'])


if __name__ == '__main__':
    main()
