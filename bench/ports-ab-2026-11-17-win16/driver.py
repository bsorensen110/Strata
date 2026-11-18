#!/usr/bin/env python3
"""Sequential A/B driver for Campaign 7 (ports-ab-2026-11-17-win16).

Supports both:
- Group A: zero-code env arms on fork-68872b2d1/strata, and
- Group B: binary + env arms comparing build-hip/strata against fork-68872b2d1/strata.

One engine owns the GPU at a time; production is stopped for the campaign window and stays
stopped on exit by default (--no-restore default True).
"""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, time, urllib.request

from gen_configs import resolve_env, resolve_exe, resolve_port

ROOT = '/home/soren/Strata-latest-20261112-182859/bench/ports-ab-2026-11-17-win16'
REPO = '/home/soren/Strata-latest-20261112-182859'
DEFAULT_EXE = '/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-68872b2d1/strata'
EXE = DEFAULT_EXE
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


def read_meminfo_gib():
    out = {}
    with open('/proc/meminfo') as fh:
        for line in fh:
            k, _, rest = line.partition(':')
            if k in ('MemTotal', 'MemFree', 'MemAvailable', 'Cached'):
                parts = rest.split()
                if parts:
                    out[k] = int(parts[0]) / (1024.0 * 1024.0)
    return out


def mem_settle_wait(min_avail_gib=48.0, timeout=30):
    time.sleep(3)
    t0 = time.monotonic()
    prev_avail = None
    mi = read_meminfo_gib()
    while time.monotonic() - t0 < timeout:
        avail = mi.get('MemAvailable', 0.0)
        if avail >= min_avail_gib and (prev_avail is None or abs(avail - prev_avail) < 0.5):
            break
        prev_avail = avail
        time.sleep(2)
        mi = read_meminfo_gib()
    log(
        f'mem settle: MemAvailable={mi.get("MemAvailable", 0.0):.2f} GiB '
        f'MemFree={mi.get("MemFree", 0.0):.2f} GiB Cached={mi.get("Cached", 0.0):.2f} GiB'
    )


def ctl_exe(a):
    return json.load(open(f'{ROOT}/configs/ctl_{a["id"]}.json'))['exe']


def trt_exe(a):
    return json.load(open(f'{ROOT}/configs/trt_{a["id"]}.json'))['exe']


def arm_label(a):
    ce, te = ctl_exe(a), trt_exe(a)
    parts = []
    if ce != te:
        parts.append(f'bin: {os.path.basename(os.path.dirname(ce))} -> {os.path.basename(os.path.dirname(te))}')
    if a.get('switch'):
        parts.append(f'{a["switch"]}: {a.get("baseline")} -> {a.get("arm")}')
    elif a.get('ctl_env') is not None or a.get('trt_env') is not None:
        ce_env = {str(k): str(v) for k, v in (a.get('ctl_env') or {}).items()}
        te_env = {str(k): str(v) for k, v in (a.get('trt_env') or {}).items()}
        keys = sorted(set(ce_env) | set(te_env))
        diffs = [f'{k}: {ce_env.get(k, "unset")}->{te_env.get(k, "unset")}' for k in keys if ce_env.get(k) != te_env.get(k)]
        if diffs:
            parts.append(', '.join(diffs))
        elif ce == te:
            parts.append('A/A identical')
        else:
            pins = [f'{k}={v}' for k, v in sorted(te_env.items())]
            if pins:
                parts.append(', '.join(pins))
    if a.get('args_ctl'):
        parts.append('ctl_args: ' + ' '.join(str(x) for x in a['args_ctl']))
    if a.get('args_arm'):
        parts.append('trt_args: ' + ' '.join(str(x) for x in a['args_arm']))
    return '; '.join(parts) or a['id']


def preflight():
    os.makedirs(f'{ROOT}/results', exist_ok=True)
    tab = json.load(open(f'{ROOT}/arms.json'))
    arms = tab['arms']
    default_exe = tab.get('default_exe') or DEFAULT_EXE
    exe_cache = {}

    def load_exe(path):
        if path not in exe_cache:
            if not os.path.isfile(path) or not os.access(path, os.X_OK):
                sys.exit(f'preflight: missing or non-executable binary: {path}')
            data = open(path, 'rb').read()
            digest = hashlib.sha256(data).hexdigest()
            exe_cache[path] = (data, digest)
            log(f'preflight exe {path} sha256={digest}')
        return exe_cache[path]

    def norm(args, sid, role):
        return [x.replace(f'{ROOT}/configs/{role}_{sid}', '<SCRATCH>') for x in args]

    seen_ports = set()
    missing = []
    for idx, a in enumerate(arms):
        sid = a['id']
        ctl_path = f'{ROOT}/configs/ctl_{sid}.json'
        trt_path = f'{ROOT}/configs/trt_{sid}.json'
        if not os.path.isfile(ctl_path) or not os.path.isfile(trt_path):
            sys.exit(f'preflight: missing config for {sid}; run gen_configs.py first')
        ctl = json.load(open(ctl_path))
        trt = json.load(open(trt_path))

        exp_ctl_exe = resolve_exe(a, 'ctl', default_exe)
        exp_trt_exe = resolve_exe(a, 'trt', default_exe)
        if ctl.get('exe') != exp_ctl_exe or trt.get('exe') != exp_trt_exe:
            sys.exit(f'preflight: {sid} exe mismatch vs arms.json')
        ctl_data, _ = load_exe(ctl['exe'])
        trt_data, _ = load_exe(trt['exe'])

        if exp_ctl_exe != exp_trt_exe and ctl['exe'] == trt['exe']:
            sys.exit(f'preflight: {sid} expects distinct binaries but both sides match')
        if (not a.get('switch') and not a.get('ctl_env') and not a.get('trt_env')
                and not sid.startswith('aa_floor') and ctl.get('exe') == trt.get('exe')):
            sys.exit(f'preflight: {sid} is a binary arm but both sides use the same engine')

        exp_ctl_port = resolve_port(a, idx, 'ctl')
        exp_trt_port = resolve_port(a, idx, 'trt')
        if ctl.get('port') != exp_ctl_port or trt.get('port') != exp_trt_port:
            sys.exit(f'preflight: {sid} ports wrong (got {ctl.get("port")}/{trt.get("port")}, expected {exp_ctl_port}/{exp_trt_port})')
        for role, p in (('ctl', ctl['port']), ('trt', trt['port'])):
            if p in seen_ports:
                sys.exit(f'preflight: duplicate port {p} on {role}_{sid}')
            seen_ports.add(p)

        if ctl.get('drafting') != 'dflash2' or trt.get('drafting') != 'dflash2':
            sys.exit(f'preflight: {sid} drafting must be "dflash2"')

        for role, cfg_obj, data in (('ctl', ctl, ctl_data), ('trt', trt, trt_data)):
            env = cfg_obj.get('env') or {}
            if env.get('STRATA_DF2_HEAD') == 'pruned':
                sys.exit(f'preflight: {role}_{sid} sets forbidden STRATA_DF2_HEAD=pruned')
            for k in env:
                if k == 'STRATA_MANIFEST':
                    continue
                if k.encode() not in data:
                    missing.append((f'{role}_{sid}', k, cfg_obj['exe']))

        exp_ctl_env = resolve_env(a, 'ctl', f'ctl_{sid}')
        exp_trt_env = resolve_env(a, 'trt', f'trt_{sid}')
        if ctl.get('env') != exp_ctl_env or trt.get('env') != exp_trt_env:
            sys.exit(f'preflight: {sid} env mismatch vs arms.json')

        for k in set(ctl) | set(trt):
            if k == 'env':
                continue
            if ctl.get(k) != trt.get(k) and k not in ('port', 'log', 'cwd', 'slot_save_path', 'args', 'exe'):
                sys.exit(f'preflight: {sid} differs on {k}')
        declares_extra_args = bool(a.get('args_arm') or a.get('args_ctl'))
        ctl_base = norm(ctl['args'], sid, 'ctl')
        trt_base = norm(trt['args'], sid, 'trt')
        exp_ctl_extra = [str(x) for x in (a.get('args_ctl') or [])]
        exp_trt_extra = [str(x) for x in (a.get('args_arm') or [])]
        if exp_ctl_extra:
            if ctl_base[-len(exp_ctl_extra):] != exp_ctl_extra:
                sys.exit(f'preflight: {sid} ctl missing args_ctl {exp_ctl_extra}')
            ctl_base = ctl_base[:-len(exp_ctl_extra)]
        if exp_trt_extra:
            if trt_base[-len(exp_trt_extra):] != exp_trt_extra:
                sys.exit(f'preflight: {sid} trt missing args_arm {exp_trt_extra}')
            trt_base = trt_base[:-len(exp_trt_extra)]
        if ctl_base != trt_base:
            sys.exit(f'preflight: {sid} engine args differ between arms' + (' beyond args_ctl/args_arm' if declares_extra_args else ''))

    if missing:
        sys.exit(f'preflight: env names absent from target engine binary: {missing}')
    log(f'preflight ok: {len(arms)} pairs, {len(seen_ports)} unique ports, {len(exe_cache)} engine binaries')


def run_arm(name, timeout=2400, suffix=''):
    port = json.load(open(f'{ROOT}/configs/{name}.json'))['port']
    unit = f'portsab-{name}'
    sh(['systemctl', '--user', 'reset-failed', unit])
    mem_settle_wait()
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
    if rr.returncode == 0 and name.split('_', 1)[1] in [a['id'] for a in json.load(open(f'{ROOT}/arms.json'))['arms'] if a.get('long')]:
        rl = sh([sys.executable, f'{ROOT}/bench_arm_long.py', name] + ([suffix] if suffix else []), timeout=timeout)
        if rl.returncode != 0:
            log(f'LONG_FAIL {name} rc={rl.returncode} :: {(rl.stdout or "")[-400:]} {(rl.stderr or "")[-500:]}')
            return 'fail'
    dt = round(time.monotonic() - t0, 1)
    sh(['systemctl', '--user', 'stop', f'{unit}.service'])
    sh(['systemctl', '--user', 'reset-failed', unit])
    rsfx = suffix or '_r1'
    if os.path.isfile(f'{ROOT}/configs/{name}/engine.log'):
        shutil.copy2(f'{ROOT}/configs/{name}/engine.log', f'{ROOT}/configs/{name}/engine{rsfx}.log')
    if rr.returncode != 0:
        log(f'ARM {name} FAIL rc={rr.returncode} {dt}s :: {(rr.stdout or "")[-600:]} {(rr.stderr or "")[-900:]}')
        return 'fail'
    tail = (rr.stdout or '').strip().splitlines()[-1] if rr.stdout.strip() else ''
    log(f'ARM {name} ok {dt}s {tail[:400]}')
    gpu_free_wait(port)
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start_at', default=None, help='resume at this arm id')
    ap.add_argument('--only', default=None, help='comma separated arm ids')
    ap.add_argument('--max-pairs', type=int, default=None)
    ap.add_argument('--skip-ctest', action='store_true')
    ap.add_argument('--round-offset', type=int, default=0, help='skip this many rounds (keeps round 1 samples intact)')
    ap.add_argument('--rounds', type=int, default=1, help='repeat each pair N times (round N saves results with a _rN suffix)')
    ap.add_argument('--no-restore', dest='no_restore', action='store_true', default=True,
                    help='leave strata.service stopped when the window ends (default: on)')
    ap.add_argument('--restore', dest='no_restore', action='store_false',
                    help='restore strata.service when the window ends')
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
    log(f'campaign start: {len(arms)} pairs, first={arms[0]["id"] if arms else None}')
    stop_prod()
    first_port = json.load(open(f'{ROOT}/configs/ctl_{arms[0]["id"]}.json'))['port'] if arms else 8300
    gpu_free_wait(first_port)
    restore = {'done': False}

    def on_exit(*_):
        if restore['done']:
            return
        restore['done'] = True
        if opts.no_restore:
            log('production left stopped (--no-restore)')
            return
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
            log(f'=== PAIR {sid} ({arm_label(a)}) ===')
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
    tr = sh([sys.executable, f'{ROOT}/telemetry.py'])
    if tr.returncode != 0:
        log(f'TELEMETRY_WARN {tr.stdout[-400:]}{tr.stderr[-400:]}')
    r = sh([sys.executable, f'{ROOT}/render_doc.py'])
    if r.returncode != 0:
        log(f'RENDER_FAIL {r.stdout[-800:]}{r.stderr[-800:]}')
    else:
        log('doc rendered: ' + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''))


if __name__ == '__main__':
    main()
