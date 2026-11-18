#!/usr/bin/env python3
"""Extract DFlash2 batch/verify telemetry from each arm's engine.log.

The engine prints, once the slots go idle: the window count, the average draft rows a window, the
timing of a window and the row throughput, plus how many drafted rows the verifier accepted. That
is the "verify width vs accepted tokens a round" view the window-width and CONF arms are tuning.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PAT = {
    'windows':   re.compile(r'strata batch: (\d+) windows, avg ([\d.]+) rows, ([\d.]+) ms/window[^\n]*?([\d.]+) rows/s'),
    'accepted':  re.compile(r'MTP proposals accepted (\d+) of (\d+) \(([\d.]+)%\)'),
    'deferred':  re.compile(r'DFlash2 slots: (\d+) rows deferred'),
    'conf_stop': re.compile(r'conf(?:idence)? stop[^\n]{0,80}'),
}


def read(name):
    p = f'{ROOT}/configs/{name}/engine.log'
    if not os.path.exists(p):
        return None
    txt = open(p, errors='replace').read()
    out = {}
    m = PAT['windows'].search(txt)
    if m:
        out['windows'] = int(m.group(1))
        out['rows_per_window'] = float(m.group(2))
        out['ms_per_window'] = float(m.group(3))
        out['rows_per_s'] = float(m.group(4))
    m = PAT['accepted'].search(txt)
    if m:
        out['accepted'] = int(m.group(1))
        out['verifier_rounds'] = int(m.group(2))
        out['accept_pct'] = float(m.group(3))
    m = PAT['deferred'].search(txt)
    if m:
        out['deferred_rows'] = int(m.group(1))
    return out or None


def main():
    tab = json.load(open(f'{ROOT}/arms.json'))
    rows = []
    for a in tab['arms']:
        sid = a['id']
        c, t = read(f'ctl_{sid}'), read(f'trt_{sid}')
        rec = {'arm': sid, 'switch': a.get('switch'), 'baseline': a['baseline'], 'arm_value': a['arm'],
               'ctl': c, 'trt': t}
        if c and t and c.get('accept_pct') and t.get('accept_pct'):
            rec['accept_pct_delta'] = round(t['accept_pct'] - c['accept_pct'], 1)
            rec['rows_per_window_ctl'], rec['rows_per_window_trt'] = c['rows_per_window'], t['rows_per_window']
            rec['ms_per_window_ctl'], rec['ms_per_window_trt'] = c['ms_per_window'], t['ms_per_window']
            # accepted tokens a round is the payload the wider window is buying; rows/s is the rate it lands at
            rec['accepted_delta'] = t['accepted'] - c['accepted']
        rows.append(rec)
    json.dump(rows, open(f'{ROOT}/results/telemetry.json', 'w'), indent=1)
    print(f"{'arm':20s} {'ctl acc%':>8s} {'trt acc%':>8s} {'Δ':>6s} {'rows/w':>7s} {'ms/w':>7s} {'rows/s':>7s}")
    for r in rows:
        c, t = r['ctl'] or {}, r['trt'] or {}
        print(f"{r['arm']:20s} {c.get('accept_pct', 0):8.1f} {t.get('accept_pct', 0):8.1f} "
              f"{r.get('accept_pct_delta', 0):+6.1f} {c.get('rows_per_window', 0):7.2f} "
              f"{t.get('ms_per_window', 0):7.2f} {t.get('rows_per_s', 0):7.1f}")
    return rows


if __name__ == '__main__':
    sys.exit(0 if main() else 0)
