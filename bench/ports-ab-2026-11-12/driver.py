#!/usr/bin/env python3
"""Sequential A/B driver for the RDNA4-Ports switch campaign.

For each switch: run the control arm (switch pinned at production's effective value), then the
treatment arm (switch at the tested value), each as a fresh engine launch, and rewrite the
results section of RDNA4-Ports.md as soon as the pair completes. One engine owns the GPU at a
time; production is stopped for the whole window and restored on exit.
"""
import argparse, json, os, subprocess, sys, time, signal, urllib.request

ROOT = '/home/soren/Strata-latest-20261112-182859/bench/ports-ab-2026-11-12'
REPO = '/home/soren/Strata-latest-20261112-182859'
EXE = f'{REPO}/build-hip/strata'
GPU_FREE = '/home/soren/.hermes/skills/software-development/strata-hip-install/scripts/gpu_free_wait.sh'
LOG = f'{ROOT}/campaign.log'


def log(msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def sh(cmd, timeout=None, check=False):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f'{cmd} -> {r.returncode}: {r.stdout[-2000:]}{r.stderr[-2000:]}')
    return r


def stop_prod():
    sh(['systemctl', '--user', 'stop', 'strata.service'])
    for _ in range(60):
        if sh(['pgrep', '-x', 'strata']).returncode != 0:
            return
        time.sleep(5)
    raise RuntimeError('other strata process still running')


def start_prod():
    sh(['systemctl', '--user', 'start', 'strata.service'])
    key = None
    for line in open('/home/soren/.config/strata-upstream-v0.2.10/production.env'):
        line = line.strip()
        if line.startswith('STRATA_API_KEY') and '=' in line:
            key = line.split('=', 1)[1].strip().strip('"').strip("'")
    prof = json.load(open('/home/soren/.local/share/strata-upstream-v0.2.10/components/strata-iq3_s.json'))
    url = f"http://127.0.0.1:{prof['port']}"
    for _ in range(240):
        try:
            q = urllib.request.Request(url + '/v1/models', headers={'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(q, timeout=5):
                return True
        except Exception:
            time.sleep(2)
    return False


def gpu_free_wait(port):
    r = sh(['bash', GPU_FREE, str(port), '600'])
    if r.returncode != 0:
        log(f'GPU_FREE_WARN {r.stdout.strip()} {r.stderr.strip()}')
        return False
    log(f'gpu free: {r.stdout.strip()}')
    return True


def preflight():
    os.makedirs(f'{ROOT}/results', exist_ok=True)
    arms = json.load(open(f'{ROOT}/arms.json'))
    data = open(EXE, 'rb').read()
    missing = [a['switch'] for a in arms['arms'] if a['switch'].encode() not in data]
    if missing:
        sys.exit(f'preflight: switch names absent from the built engine: {missing}')
    # paired configs must differ only in env (scratch paths normalized: both point at their own arm dir)
    def norm(args, sid, role):
        return [x.replace(f'{ROOT}/configs/{role}_{sid}', '<SCRATCH>') for x in args]
    for a in arms['arms']:
        ctl = json.load(open(f'{ROOT}/configs/ctl_{a["id"]}.json'))
        trt = json.load(open(f'{ROOT}/configs/trt_{a["id"]}.json'))
        for k in set(ctl) | set(trt):
            if k == 'env':
                continue
            if ctl.get(k) != trt.get(k) and k not in ('port', 'log', 'cwd', 'slot_save_path', 'args'):
                sys.exit(f'preflight: {a["id"]} differs on {k}')
        if norm(ctl['args'], a['id'], 'ctl') != norm(trt['args'], a['id'], 'trt'):
            sys.exit(f'preflight: {a["id"]} engine args differ between arms')
        if ctl['port'] != 8300 or trt['port'] != 8301:
            sys.exit(f'preflight: {a["id"]} ports wrong')
        if ctl['env'].get(a['switch']) != str(a['baseline']) or trt['env'].get(a['switch']) != str(a['arm']):
            sys.exit(f'preflight: {a["id"]} arm env does not pin the switch')
    log(f'preflight ok: {len(arms["arms"])} switches, binary {os.stat(EXE).st_size} bytes')


def run_arm(name, timeout=2400, suffix=''):
    port = json.load(open(f'{ROOT}/configs/{name}.json'))['port']
    unit = f'portsab-{name}'
    sh(['systemctl', '--user', 'reset-failed', unit])
    # stale startup text from an earlier launch (e.g. an OOM from a collided run) must not be
    # mistaken for this launch failing
    for f in (f'{ROOT}/configs/{name}/engine.log', f'{ROOT}/configs/{name}/serve.out'):
        open(f, 'w').close()
    r = sh(['bash', f'{ROOT}/start-arm.sh', name])
    if r.returncode != 0:
        log(f'ARM {name} LAUNCH_FAIL {r.stdout[-800:]} {r.stderr[-800:]}')
        return 'launch_fail'
    t0 = time.monotonic()
    rr = sh([sys.executable, f'{ROOT}/bench_arm.py', name] + ([suffix] if suffix else []), timeout=timeout)
    dt = round(time.monotonic() - t0, 1)
    sh(['systemctl', '--user', 'stop', f'{unit}.service'])
    sh(['systemctl', '--user', 'reset-failed', unit])
    if rr.returncode != 0:
        log(f'ARM {name} FAIL rc={rr.returncode} {dt}s :: {(rr.stdout or "")[-600:]} {(rr.stderr or "")[-900:]}')
        return 'fail'
    tail = (rr.stdout or '').strip().splitlines()[-1] if rr.stdout.strip() else ''
    log(f'ARM {name} ok {dt}s {tail[:400]}')
    gpu_free_wait(port)
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start_at', default=None, help='resume at this switch id')
    ap.add_argument('--only', default=None, help='comma separated switch ids')
    ap.add_argument('--max-pairs', type=int, default=None)
    ap.add_argument('--skip-ctest', action='store_true')
    ap.add_argument('--round-offset', type=int, default=0, help='skip this many rounds (keeps round 1 samples intact)')
    ap.add_argument('--rounds', type=int, default=1, help='repeat each pair N times (round N saves results with a _rN suffix)')
    opts = ap.parse_args()

    arms = json.load(open(f'{ROOT}/arms.json'))['arms']
    ids = [a['id'] for a in arms]
    if opts.only:
        want = [x.strip() for x in opts.only.split(',')]
        bad = [x for x in want if x not in ids]
        if bad:
            sys.exit(f'unknown ids {bad}')
        arms = [a for a in arms if a['id'] in want]
    elif opts.start_at:
        arms = arms[ids.index(opts.start_at):]
    if opts.max_pairs:
        arms = arms[:opts.max_pairs]

    preflight()
    log(f'campaign start: {len(arms)} switch pairs, first={arms[0]["id"] if arms else None}')
    stop_prod()
    gpu_free_wait(8300)
    restore = {'done': False}

    def on_exit(*_):
        if restore['done']:
            return
        restore['done'] = True
        log('restoring production service')
        ok = start_prod()
        log(f'production restored: {"ready" if ok else "NOT READY"}')

    signal.signal(signal.SIGTERM, lambda *a: (on_exit(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *a: (on_exit(), sys.exit(0)))

    try:
        if not opts.skip_ctest:
            log('ctest parity subset')
            r = sh(['ctest', '--test-dir', f'{REPO}/build-hip', '-R',
                    '^(pf_wmma_parity|rope_parity|kv_steps_parity|mmvq_il_parity_pf|gdn_parity|'
                    'shared_expert_parity|native_multi_parity|qsa_merge_daily_parity_default)$',
                    '--output-on-failure'], timeout=3600)
            with open(f'{ROOT}/results/ctest.log', 'w') as fh:
                fh.write(r.stdout + r.stderr)
            summary = [l for l in r.stdout.splitlines() if 'tests passed' in l or 'Failed' in l]
            log(f'ctest rc={r.returncode} {summary[-1] if summary else ""}')

        for a in arms:
            sid = a['id']
            log(f'=== PAIR {sid} ({a["switch"]}: prod {a["baseline"]} -> arm {a["arm"]}) ===')
            for rnd in range(1 + opts.round_offset, 1 + opts.round_offset + opts.rounds):
                sfx = '' if rnd == 1 else f'_r{rnd}'
                ctl = run_arm(f'ctl_{sid}', suffix=sfx)
                if ctl != 'ok':
                    log(f'PAIR {sid} round {rnd} skipped: control arm {ctl}')
                    break
                trt = run_arm(f'trt_{sid}', suffix=sfx)
                log(f'PAIR {sid} round {rnd} control={ctl} treatment={trt}')
            render()
    finally:
        on_exit()


def render():
    r = sh([sys.executable, f'{ROOT}/render_doc.py'])
    if r.returncode != 0:
        log(f'RENDER_FAIL {r.stdout[-800:]}{r.stderr[-800:]}')
    else:
        log('doc rendered: ' + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''))


if __name__ == '__main__':
    main()
