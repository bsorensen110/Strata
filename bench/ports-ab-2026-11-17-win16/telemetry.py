#!/usr/bin/env python3
"""Extract DFlash2 batch/verify and KFD-pause telemetry across round logs for each arm."""
import glob, json, os, re, statistics as st, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PAT = {
    'windows':   re.compile(r'strata batch: (\d+) windows, avg ([\d.]+) rows, ([\d.]+) ms/window[^\n]*?([\d.]+) rows/s'),
    'run_split': re.compile(r'= run ([\d.]+) \(CUDA\d+ GPU-reach wait ([\d.]+) \+ CPU experts ([\d.]+)\)'),
    'accepted':  re.compile(r'MTP proposals accepted (\d+) of (\d+) \(([\d.]+)%\)'),
    'deferred':  re.compile(r'DFlash2 slots: (\d+) rows deferred'),
    'gpu_pause': re.compile(r'paused the GPU for (\d+) ms'),
}


def parse_log(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    txt = open(path, errors='replace').read()
    out = {}
    m = PAT['windows'].search(txt)
    if m:
        out['windows'] = int(m.group(1))
        out['rows_per_window'] = float(m.group(2))
        out['ms_per_window'] = float(m.group(3))
        out['rows_per_s'] = float(m.group(4))
    m = PAT['run_split'].search(txt)
    if m:
        out['bt_run'] = float(m.group(1))
        out['gpu_wait'] = float(m.group(2))
        out['cpu_exp'] = float(m.group(3))
        out['other_ms'] = round(out['bt_run'] - (out['gpu_wait'] + out['cpu_exp']), 2)
    m = PAT['accepted'].search(txt)
    if m:
        out['accepted'] = int(m.group(1))
        out['verifier_rounds'] = int(m.group(2))
        out['accept_pct'] = float(m.group(3))
    m = PAT['deferred'].search(txt)
    if m:
        out['deferred_rows'] = int(m.group(1))
    pauses = [int(x) for x in PAT['gpu_pause'].findall(txt)]
    if out:
        out['gpu_pause_count'] = len(pauses)
        out['gpu_pause_ms'] = sum(pauses)
    return out or None


def read(name):
    cdir = f'{ROOT}/configs/{name}'
    rlogs = []
    for p in glob.glob(f'{cdir}/engine_r*.log'):
        m = re.search(r'engine_r(\d+)\.log$', p)
        if m:
            rlogs.append((int(m.group(1)), p))
    rlogs.sort()
    paths = [p for _, p in rlogs] if rlogs else [f'{cdir}/engine.log']
    parsed = []
    for idx, p in enumerate(paths, 1):
        rec = parse_log(p)
        if rec:
            rec['log_round'] = rlogs[idx - 1][0] if rlogs else 1
            parsed.append(rec)
    if not parsed:
        return None
    agg = {'rounds': len(parsed), 'by_round': parsed}
    for k in ('windows', 'rows_per_window', 'ms_per_window', 'rows_per_s',
              'bt_run', 'gpu_wait', 'cpu_exp', 'other_ms',
              'accepted', 'verifier_rounds', 'accept_pct', 'deferred_rows'):
        vals = [x[k] for x in parsed if k in x]
        if vals:
            v = st.median(vals)
            agg[k] = int(round(v)) if isinstance(vals[0], int) and len(vals) % 2 == 1 else round(float(v), 2)
    agg['gpu_pause_count'] = sum(x.get('gpu_pause_count', 0) for x in parsed)
    agg['gpu_pause_ms'] = sum(x.get('gpu_pause_ms', 0) for x in parsed)
    return agg


def main():
    tab = json.load(open(f'{ROOT}/arms.json'))
    rows = []
    for a in tab['arms']:
        sid = a['id']
        c, t = read(f'ctl_{sid}'), read(f'trt_{sid}')
        rec = {
            'arm': sid,
            'group': a.get('group'),
            'switch': a.get('switch'),
            'baseline': a.get('baseline'),
            'arm_value': a.get('arm'),
            'ctl_env': a.get('ctl_env'),
            'trt_env': a.get('trt_env'),
            'ctl_exe': a.get('ctl_exe') or a.get('exe_ctl'),
            'trt_exe': a.get('trt_exe') or a.get('exe_arm'),
            'ctl': c,
            'trt': t,
        }
        if c and t and c.get('accept_pct') is not None and t.get('accept_pct') is not None:
            rec['accept_pct_delta'] = round(t['accept_pct'] - c['accept_pct'], 1)
            rec['rows_per_window_ctl'], rec['rows_per_window_trt'] = c.get('rows_per_window'), t.get('rows_per_window')
            rec['ms_per_window_ctl'], rec['ms_per_window_trt'] = c.get('ms_per_window'), t.get('ms_per_window')
            rec['accepted_delta'] = round(t['accepted'] - c['accepted'], 1)
        rows.append(rec)
    os.makedirs(f'{ROOT}/results', exist_ok=True)
    with open(f'{ROOT}/results/telemetry.json', 'w') as fh:
        json.dump(rows, fh, indent=1)
    print(f"{'arm':28s} {'ctl acc%':>8s} {'trt acc%':>8s} {'d':>6s} {'ms/w c->t':>13s} {'other c->t':>13s} {'pause c/t':>9s}")
    for r in rows:
        c, t = r['ctl'] or {}, r['trt'] or {}
        print(f"{r['arm']:28s} {c.get('accept_pct', 0):8.1f} {t.get('accept_pct', 0):8.1f} "
              f"{r.get('accept_pct_delta', 0):+6.1f} "
              f"{c.get('ms_per_window', 0):5.1f}->{t.get('ms_per_window', 0):5.1f}  "
              f"{c.get('other_ms', 0):5.2f}->{t.get('other_ms', 0):5.2f}  "
              f"{c.get('gpu_pause_count', 0):3d}/{t.get('gpu_pause_count', 0):<3d}")
    return rows


if __name__ == '__main__':
    sys.exit(0 if main() else 0)
