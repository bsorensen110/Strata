#!/usr/bin/env python3
"""IL on/off A/B, v5 (candidate build, --mmap-experts).

  S0 / S1  single slot, MTP drafter, IL off / on
  K0 / K1  batch MTP (--batch 4 --batch-mtp, the production layout), IL off / on;
           one request is timed at a time

Each round: 3 warmup requests (128 tokens), then 10 measured requests (512 tokens, greedy).
Order: S0 S1 | S1 S0 | K0 K1 | K1 K0. Production stays stopped until every benchmark process
has exited; it is restarted only after that.
"""
import json
import os
import random
import signal
import statistics as st
import subprocess
import time
import urllib.request

D = '/home/soren/Strata-ab/il-decode-ab-20261028'
PROD_CFG = '/home/soren/Strata/strata-swift-iq3_s.json'
CAND = '/home/soren/Strata-integ/build-hip/strata'
PY = '/home/soren/Strata/.venv/bin/python'
TEST = 'http://127.0.0.1:8086'
PROD = 'http://192.168.1.156:8086'
MIN_AVAIL_KB = 30 * 1024 * 1024
N_WARM, N_MEAS, MAX_TOK = 3, 10, 512

os.umask(0o077)
cfg0 = json.load(open(PROD_CFG))
KEY = cfg0['api_key']


def drop_pair(args, flag):
    while flag in args:
        i = args.index(flag)
        del args[i:i + 2]


def arm_cfg(name):
    single = name.startswith('S')
    il = '1' if name.endswith('1') else '0'
    c = json.loads(json.dumps(cfg0))
    c['host'], c['port'] = '127.0.0.1', 8086
    c['exe'] = CAND
    c['log'] = f'{D}/engine-{name}.log'
    for f in ('--conversation-cache-disk-mib', '--conversation-cache-dir', '--kv-pool-tokens'):
        drop_pair(c['args'], f)
    if '--mmap-experts' not in c['args']:
        c['args'].append('--mmap-experts')
    env = c.setdefault('env', {})
    env.update({'STRATA_PREFILL_CPU_SHARE': '0', 'STRATA_ROUTE_TAIL_SKIP': '0',
                'STRATA_DECODE_TIMING': '1', 'STRATA_MMVQ_IL': il})
    if single:
        c['parallel'] = 1
        c['args'] = [a for a in c['args'] if a != '--batch-mtp']
        drop_pair(c['args'], '--batch')
        c['drafting'] = 'mtp'
    else:
        c['parallel'] = 4
    c.pop('api_key', None)            # the key goes only in the server environment
    p = f'{D}/config-{name}.json'
    with open(p, 'w') as fh:
        json.dump(c, fh, indent=1)
    os.chmod(p, 0o600)
    return c


def prompts(rnd):
    base = 10000 * (rnd - 1)
    return [f'List every integer from {300 + 250 * i + base} to {549 + 250 * i + base} in order, one per line.'
            for i in range(N_MEAS)]


def http_model(base):
    try:
        q = urllib.request.Request(base + '/v1/models', headers={'Authorization': 'Bearer ' + KEY})
        with urllib.request.urlopen(q, timeout=5) as r:
            return json.loads(r.read())['data'][0]['id']
    except Exception:
        return None


def chat(model, prompt, maxt):
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': maxt, 'temperature': 0.0,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    q = urllib.request.Request(TEST + '/v1/chat/completions', data=body,
                               headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(q, timeout=900) as r:
        d = json.loads(r.read())
    u = d['usage']
    tm = d.get('timings') or {}
    return {'ct': u['completion_tokens'], 'finish': d['choices'][0].get('finish_reason'),
            'engine_tps': tm.get('predicted_per_second'),
            'timings': {k: v for k, v in tm.items() if isinstance(v, (int, float, str))}}


def mem_avail_kb():
    with open('/proc/meminfo') as fh:
        for line in fh:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])
    return 0


def _argv(pid):
    try:
        raw = open(f'/proc/{pid}/cmdline', 'rb').read()
    except Exception:
        return []
    return [x.decode(errors='replace') for x in raw.split(b'\0') if x]


def _ppid(pid):
    try:
        return int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
    except Exception:
        return -1


def test_procs():
    """Benchmark-owned processes only: the candidate engine and serve.server on a bench config."""
    me = os.getpid()
    found = []
    for e in os.listdir('/proc'):
        if not e.isdigit() or int(e) == me:
            continue
        av = _argv(int(e))
        if not av:
            continue
        if av[0] == CAND:
            found.append(int(e))
        elif av[0] == PY and 'serve.server' in av and any('/il-decode-ab-20261028/config-' in x for x in av):
            found.append(int(e))
    return found


def stop_arm(pr):
    if pr.poll() is None:
        try:
            os.kill(pr.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    t0 = time.time()
    while time.time() - t0 < 150 and (pr.poll() is None or test_procs()):
        time.sleep(2)
    for p in test_procs():
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        pr.wait(timeout=15)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < 120 and mem_avail_kb() < MIN_AVAIL_KB:
        time.sleep(3)


def run_arm(name, roundi):
    rec = {'arm': name, 'round': roundi}
    avail = mem_avail_kb()
    rec['mem_avail_gib_before'] = round(avail / 1048576, 1)
    if avail < MIN_AVAIL_KB:
        rec['error'] = 'skipped: MemAvailable below 30 GiB before start'
        return rec
    c = arm_cfg(name)
    log_path = c['log']
    off = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    out = open(f'{D}/serve-{name}-{roundi}.out', 'w')
    pr = subprocess.Popen([PY, '-u', '-m', 'serve.server', '--engine', 'strata',
                           '--config', f'{D}/config-{name}.json', '--host', '127.0.0.1', '--port', '8086'],
                          cwd='/home/soren/Strata', env={**os.environ, 'STRATA_API_KEY': KEY},
                          stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        model = None
        while time.time() - t0 < 900 and pr.poll() is None:
            model = http_model(TEST)
            if model:
                break
            time.sleep(3)
        rec['ready_s'] = round(time.time() - t0, 1)
        if not model:
            rec['error'] = f'not ready (serve exit {pr.poll()})'
            return rec
        eng = [p for p in test_procs() if _ppid(p) == pr.pid]
        if eng:
            rec['eng_argv'] = _argv(eng[0])
            envz = open(f'/proc/{eng[0]}/environ', 'rb').read().split(b'\0')
            rec['eng_env'] = [x.decode(errors='replace') for x in envz
                              if x.startswith((b'STRATA_MMVQ_IL=', b'STRATA_DECODE_TIMING=',
                                               b'STRATA_PREFILL_CPU_SHARE=', b'STRATA_ROUTE_TAIL_SKIP='))]
        for _ in range(N_WARM):
            chat(model, 'Count upward from 88000 for 100 numbers, one per line.', 128)
        rows = [chat(model, p, MAX_TOK) for p in prompts(roundi)]
        ok = [r['engine_tps'] for r in rows if r['ct'] >= 500 and r['finish'] == 'length' and r['engine_tps']]
        rec['engine_tps'] = [round(r['engine_tps'], 2) if r['engine_tps'] else None for r in rows]
        rec['ok_tps'] = [round(x, 2) for x in ok]
        rec['completion_tokens'] = [r['ct'] for r in rows]
        rec['finish'] = [r['finish'] for r in rows]
        rec['timings_first'] = rows[0]['timings']
        rec['n_ok'] = len(ok)
        rec['median_tps'] = round(st.median(ok), 2) if len(ok) >= 5 else None
    except Exception as ex:
        rec['error'] = f'{type(ex).__name__}: {ex}'
    finally:
        stop_arm(pr)
        rec['test_procs_after_stop'] = test_procs()
        out.close()
    return rec


def prod_ready():
    t0 = time.time()
    while time.time() - t0 < 420:
        if http_model(PROD):
            return True
        time.sleep(3)
    return False


def boot_ci(off_vals, on_vals, n=4000, seed=1):
    """95% bootstrap CI of (median on - median off), tok/s."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        ra = [rng.choice(off_vals) for _ in off_vals]
        rb = [rng.choice(on_vals) for _ in on_vals]
        diffs.append(st.median(rb) - st.median(ra))
    diffs.sort()
    return round(diffs[int(0.025 * n)], 2), round(diffs[int(0.975 * n)], 2)


def welch_t(off_vals, on_vals):
    ma, mb = st.mean(off_vals), st.mean(on_vals)
    va, vb = st.variance(off_vals), st.variance(on_vals)
    return round((mb - ma) / ((va / len(off_vals) + vb / len(on_vals)) ** 0.5), 2)


def summarize(recs):
    def vals(arm, rnd=None):
        out = []
        for r in recs:
            if r['arm'] == arm and (rnd is None or r['round'] == rnd):
                out += r.get('ok_tps') or []
        return out

    res = {}
    for a in ('S0', 'S1', 'K0', 'K1'):
        v = vals(a)
        res[a] = {'n': len(v),
                  'median_all': round(st.median(v), 2) if v else None,
                  'mean_all': round(st.mean(v), 2) if v else None,
                  'min': min(v) if v else None, 'max': max(v) if v else None,
                  'per_round_median': [round(st.median(vals(a, k)), 2) if vals(a, k) else None for k in (1, 2)],
                  'below_0.8x_median': sorted(x for x in v if v and x < 0.8 * st.median(v))}
    out = {'arms': res}
    for label, off, on in (('single_slot_MTP', 'S0', 'S1'), ('batch_MTP', 'K0', 'K1')):
        off_v, on_v = vals(off), vals(on)
        block = {'delta_median_pct': None, 'delta_median_tps': None, 'bootstrap95_median_delta_tps': None,
                 'welch_t': None, 'per_round_pct_on_vs_off': None}
        if len(off_v) >= 5 and len(on_v) >= 5:
            mo, mn = st.median(off_v), st.median(on_v)
            block['delta_median_tps'] = round(mn - mo, 2)
            block['delta_median_pct'] = round((mn / mo - 1) * 100, 2)
            block['bootstrap95_median_delta_tps'] = boot_ci(off_v, on_v)
            block['welch_t'] = welch_t(off_v, on_v)
            pr = []
            for k in (1, 2):
                a, b = res[off]['per_round_median'][k - 1], res[on]['per_round_median'][k - 1]
                pr.append(round((b / a - 1) * 100, 2) if a and b else None)
            block['per_round_pct_on_vs_off'] = pr
        out[label] = block
    with open(f'{D}/summary-v5.json', 'w') as fh:
        json.dump({'summary': out, 'records': recs}, fh, indent=1)
    print(json.dumps(out, indent=1), flush=True)


def main():
    if test_procs():
        print('ABORT: benchmark processes already alive', test_procs(), flush=True)
        return
    subprocess.run(['systemctl', '--user', 'stop', 'strata.service'])
    time.sleep(2)
    order = [('S0', 1), ('S1', 1), ('S1', 2), ('S0', 2), ('K0', 1), ('K1', 1), ('K1', 2), ('K0', 2)]
    recs = []
    led = open(f'{D}/arms-v5.jsonl', 'w')
    try:
        for arm, rnd in order:
            r = run_arm(arm, rnd)
            recs.append(r)
            led.write(json.dumps(r) + '\n')
            led.flush()
            print(f"{arm} r{rnd}: median={r.get('median_tps')} n_ok={r.get('n_ok')} "
                  f"ready={r.get('ready_s')}s err={r.get('error')}", flush=True)
    finally:
        led.close()
        left = test_procs()
        if left:
            print('production NOT restarted: benchmark processes alive', left, flush=True)
        else:
            subprocess.run(['systemctl', '--user', 'start', 'strata.service'])
            print('production restored, authenticated ready:', prod_ready(), flush=True)
    summarize(recs)


if __name__ == '__main__':
    main()
