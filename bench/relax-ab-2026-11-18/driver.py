#!/usr/bin/env python3
"""Sequential A/B driver for the RDNA4-Ports switch campaign.

Run each control/treatment pair in interleaved rounds. Production must already be stopped before
the campaign starts and is restored only after all campaign-owned services and resources clear.
"""
import argparse, json, os, re, subprocess, sys, time, signal, urllib.error, urllib.request
import gen_configs

ROOT = '/home/soren/scratch/Strata-merge-0161/bench/relax-ab-2026-11-18'
REPO = '/home/soren/scratch/Strata-merge-0161'
EXE = f'{REPO}/build-hip-port/strata'
GPU_FREE = '/home/soren/.hermes/skills/software-development/strata-hip-install/scripts/gpu_free_wait.sh'
LOG = f'{ROOT}/campaign.log'


def campaign_arm_names():
    arms = json.load(open(f'{ROOT}/arms.json'))['arms']
    return tuple(f'{role}_{arm["id"]}' for arm in arms for role in ('ctl', 'trt'))


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


def start_prod():
    key = None
    for line in open('/home/soren/.config/strata-upstream-v0.2.10/production.env'):
        line = line.strip()
        if line.startswith('STRATA_API_KEY') and '=' in line:
            key = line.split('=', 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError('production API key missing from production.env')
    prof = json.load(open('/home/soren/.local/share/strata-upstream-v0.2.10/components/strata-iq3_s.json'))
    url = f"http://127.0.0.1:{prof['port']}"
    sh(['systemctl', '--user', 'start', 'strata.service'], check=True)
    last_error = None
    for _ in range(240):
        try:
            q = urllib.request.Request(url + '/v1/models', headers={'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(q, timeout=5):
                return True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            time.sleep(2)
    log(f'production readiness timeout: {last_error}')
    return False


def system_memory_wait(timeout=600):
    # Production uses ~47 GiB RAM with this profile; do not overlap even a candidate
    # initialization until host allocations from the previous engine have been released.
    minimum_kib = 52 * 1024 * 1024
    deadline = time.monotonic() + timeout
    last_kib = 0
    while time.monotonic() < deadline:
        with open('/proc/meminfo') as fh:
            last_kib = int(next(line.split()[1] for line in fh if line.startswith('MemAvailable:')))
        if last_kib >= minimum_kib:
            log(f'system memory free: MemAvailable {last_kib / 1024 / 1024:.2f} GiB (minimum 52 GiB)')
            return True
        time.sleep(5)
    log(f'SYSTEM_MEMORY_WAIT_TIMEOUT: MemAvailable {last_kib / 1024 / 1024:.2f} GiB < 52 GiB')
    return False


def gpu_free_wait(port):
    r = sh(['bash', GPU_FREE, str(port), '600'])
    if r.returncode != 0:
        log(f'GPU_FREE_WARN {r.stdout.strip()} {r.stderr.strip()}')
        return False
    log(f'gpu free: {r.stdout.strip()}')
    return True


def resources_free_wait(port):
    gpu_free = gpu_free_wait(port)
    memory_free = system_memory_wait()
    return gpu_free and memory_free


def production_state():
    r = sh(['systemctl', '--user', 'is-active', 'strata.service'])
    state = r.stdout.strip()
    if state == 'active':
        return state
    if r.returncode not in (0, 3):
        raise RuntimeError(f'could not verify production service state: {r.stdout.strip()} {r.stderr.strip()}')
    return state


def require_production_inactive(where):
    state = production_state()
    if state not in ('inactive', 'failed'):
        raise RuntimeError(f'{where}: production strata.service state is {state!r}, not safely inactive')
    log(f'production service inactive: {where} ({state or "unknown"})')


def result_status(name, suffix, state, reason='', expected_slots=None, actual_slots=None,
                  requested_budget=None):
    path = f'{ROOT}/results/{name}{suffix}.status.json'
    record = {'arm': name, 'round': suffix or 'r1', 'status': state,
              'reason': reason, 'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if expected_slots is not None:
        record['expected_expert_cache_slots'] = expected_slots
        record['actual_expert_cache_slots'] = actual_slots
    if requested_budget is not None:
        record['expert_cache_budget_slots'] = requested_budget
    with open(path, 'w') as fh:
        json.dump(record, fh, indent=1)
    if expected_slots is not None or requested_budget is not None:
        log(f'ARM_STATUS arm={name} status={state} expected_actual={expected_slots} '
            f'budget={requested_budget} actual={actual_slots} reason={reason}')


def cache_gate_failure(output):
    match = re.search(
        r'^CACHE_GATE_FAIL kind=(missing|mismatch) expected_actual=(\d+) '
        r'actual=(none|\d+) budget=(\d+)$', output, re.MULTILINE)
    if not match:
        return None
    kind, expected, actual, budget = match.groups()
    return {
        'status': 'cache_evidence_missing' if kind == 'missing' else 'cache_mismatch',
        'expected': int(expected),
        'actual': int(actual) if actual != 'none' else None,
        'budget': int(budget),
        'reason': match.group(0),
    }


def stop_campaign_units():
    ok = True
    names = campaign_arm_names()
    for name in names:
        unit = f'portsab-{name}.service'
        try:
            sh(['systemctl', '--user', 'stop', unit], timeout=60)
        except subprocess.TimeoutExpired:
            log(f'CLEANUP_STOP_TIMEOUT {unit}')
            ok = False
    for name in names:
        unit = f'portsab-{name}.service'
        r = sh(['systemctl', '--user', 'is-active', unit])
        state = r.stdout.strip()
        if state in ('active', 'activating', 'deactivating', 'reloading'):
            log(f'CLEANUP_UNIT_STILL_ACTIVE {unit}')
            ok = False
        elif state not in ('inactive', 'failed', 'unknown') or r.returncode not in (0, 3, 4):
            log(f'CLEANUP_UNIT_STATE_UNKNOWN {unit}: {r.stdout.strip()} {r.stderr.strip()}')
            ok = False
    return ok


def all_resources_free_wait():
    ok_8300 = gpu_free_wait(8300)
    ok_8301 = gpu_free_wait(8301)
    memory_free = system_memory_wait()
    return ok_8300 and ok_8301 and memory_free


def ctl_exe(a):
    return json.load(open(f'{ROOT}/configs/ctl_{a["id"]}.json'))['exe']


def trt_exe(a):
    return json.load(open(f'{ROOT}/configs/trt_{a["id"]}.json'))['exe']


def preflight():
    os.makedirs(f'{ROOT}/results', exist_ok=True)
    paths = gen_configs.assert_preflight()
    log(f'config preflight ok: {len(paths)} unique dispatch reports')
    arms = json.load(open(f'{ROOT}/arms.json'))
    missing = []
    for a in arms['arms']:
        if not a.get('switch'):
            continue                      # binary arms pin no switch
        data = open(a.get('exe') or EXE, 'rb').read()
        if a['switch'].encode() not in data:
            missing.append((a['id'], a['switch']))
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
            if ctl.get(k) != trt.get(k) and k not in ('port', 'log', 'cwd', 'slot_save_path', 'args', 'exe'):
                sys.exit(f'preflight: {a["id"]} differs on {k}')
        declares_extra_args = bool(a.get('args_arm') or a.get('args_ctl'))
        if not declares_extra_args and norm(ctl['args'], a['id'], 'ctl') != norm(trt['args'], a['id'], 'trt'):
            sys.exit(f'preflight: {a["id"]} engine args differ between arms')
        if ctl['port'] != 8300 or trt['port'] != 8301:
            sys.exit(f'preflight: {a["id"]} ports wrong')
        for role, cfg in (('ctl', ctl), ('trt', trt)):
            user_args = cfg.get('user_args') or []
            for flag, value in (('--lookup-max-t', '16'), ('--lookup-sizes', '10,12,14,16'),
                                ('--lookup-chain', '6'), ('--spec', '8'), ('--draft-conf', '0.35')):
                positions = [arg_pos for arg_pos, arg in enumerate(user_args) if arg == flag]
                if len(positions) != 1:
                    sys.exit(f'preflight: {role}_{a["id"]} missing production user arg {flag} {value}')
                arg_pos = positions[0]
                if arg_pos + 1 >= len(user_args) or user_args[arg_pos + 1] != value:
                    sys.exit(f'preflight: {role}_{a["id"]} production user arg changed: {flag}')
        if a.get('switch') and (ctl['env'].get(a['switch']) != str(a['baseline']) or trt['env'].get(a['switch']) != str(a['arm'])):
            sys.exit(f'preflight: {a["id"]} arm env does not pin the switch')
        if not a.get('switch') and ctl.get('exe') == trt.get('exe'):
            sys.exit(f'preflight: {a["id"]} is a binary arm but both sides use the same engine')
    log(f'preflight ok: {len(arms["arms"])} pairs')


def run_arm(name, timeout=2400, suffix=''):
    cfg = json.load(open(f'{ROOT}/configs/{name}.json'))
    port = cfg['port']
    report_path = cfg.get('env', {}).get('STRATA_DISPATCH_REPORT')
    arm_id = name.removeprefix('ctl_').removeprefix('trt_')
    arms = json.load(open(f'{ROOT}/arms.json'))['arms']
    expected_slots = next(a['expert_cache_slots'] for a in arms if a['id'] == arm_id)
    requested_budget = next(a['expert_cache_budget_slots'] for a in arms if a['id'] == arm_id)
    actual_slots = None
    try:
        require_production_inactive(f'before {name}')
    except RuntimeError as e:
        log(f'ARM {name} REFUSED: {e}')
        result_status(name, suffix, 'production_active', str(e), expected_slots,
                      requested_budget=requested_budget)
        return 'production_active'
    if not resources_free_wait(port):
        log(f'ARM {name} REFUSED: GPU/host memory not free before launch')
        result_status(name, suffix, 'resources_busy', 'GPU/port or MemAvailable was not free before launch',
                      expected_slots, requested_budget=requested_budget)
        return 'resources_busy'
    unit = f'portsab-{name}'
    sh(['systemctl', '--user', 'reset-failed', unit])
    result_status(name, suffix, 'running', expected_slots=expected_slots,
                  requested_budget=requested_budget)
    # stale startup text from an earlier launch (e.g. an OOM from a collided run) must not be
    # mistaken for this launch failing
    for f in (f'{ROOT}/configs/{name}/engine.log', f'{ROOT}/configs/{name}/serve.out'):
        open(f, 'w').close()
    if report_path and os.path.exists(report_path):
        os.unlink(report_path)
    outcome = 'fail'
    reason = ''
    t0 = time.monotonic()
    try:
        r = sh(['bash', f'{ROOT}/start-arm.sh', name])
        if r.returncode != 0:
            reason = f'launch failed: {r.stdout[-800:]} {r.stderr[-800:]}'
            log(f'ARM {name} LAUNCH_FAIL {reason}')
            outcome = 'launch_fail'
        else:
            rr = sh([sys.executable, f'{ROOT}/bench_arm.py', name] + ([suffix] if suffix else []), timeout=timeout)
            if rr.returncode != 0:
                output = (rr.stdout or '') + '\n' + (rr.stderr or '')
                gate = cache_gate_failure(output)
                if gate:
                    outcome = gate['status']
                    reason = gate['reason']
                    expected_slots = gate['expected']
                    requested_budget = gate['budget']
                    actual_slots = gate['actual']
                    log(f'ARM {name} CACHE_GATE_FAIL {reason}')
                else:
                    reason = f'bench rc={rr.returncode}: {(rr.stdout or "")[-600:]} {(rr.stderr or "")[-900:]}'
                    log(f'ARM {name} FAIL {reason}')
            else:
                result_file = f'{ROOT}/results/{name}{suffix}.json'
                rec = json.load(open(result_file))
                actual_slots = rec.get('expert_cache_slots')
                if rec.get('timing_valid') is not True:
                    reason = rec.get('invalid_reason') or 'route evidence did not match the arm configuration'
                    log(f'ARM {name} ROUTE_INVALID: {reason}')
                    evidence = rec.get('route_evidence') or {}
                    debug = {
                        'arm': name,
                        'round': suffix or 'r1',
                        'dispatch_report': report_path,
                        'launches': evidence.get('dispatch_report_launches', {}),
                        'counts_first_20': evidence.get('dispatch_report_counts_preview', []),
                    }
                    with open(f'{ROOT}/campaign.out', 'a') as fh:
                        fh.write(json.dumps(debug) + '\n')
                    outcome = 'route_invalid'
                else:
                    tail = (rr.stdout or '').strip().splitlines()[-1] if rr.stdout.strip() else ''
                    log(f'ARM {name} ok {round(time.monotonic() - t0, 1)}s {tail[:400]}')
                    outcome = 'ok'
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        reason = f'{type(e).__name__}: {e}'
        log(f'ARM {name} ERROR: {reason}')
    finally:
        state = sh(['systemctl', '--user', 'is-active', f'{unit}.service'])
        stop_ok = state.stdout.strip() in ('inactive', 'failed', 'unknown') and state.returncode in (0, 3, 4)
        if state.stdout.strip() in ('active', 'activating', 'deactivating', 'reloading'):
            try:
                stopped = sh(['systemctl', '--user', 'stop', f'{unit}.service'], timeout=60)
                stop_ok = stopped.returncode == 0
            except subprocess.TimeoutExpired:
                stop_ok = False
        sh(['systemctl', '--user', 'reset-failed', unit])
        resources_free = resources_free_wait(port)
        if not stop_ok:
            reason = (reason + '; ' if reason else '') + 'candidate stop command failed or timed out'
            outcome = 'cleanup_failed'
        if not resources_free:
            reason = (reason + '; ' if reason else '') + 'GPU/port or MemAvailable did not clear'
            outcome = 'resources_busy'
        result_status(name, suffix, 'completed' if outcome == 'ok' else outcome, reason,
                      expected_slots, actual_slots, requested_budget)
    return outcome


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
    require_production_inactive('campaign start')
    restore = {'done': False}

    def on_exit(*_):
        if restore['done']:
            return
        restore['done'] = True
        stopped = stop_campaign_units()
        resources_free = all_resources_free_wait()
        if not stopped:
            log('PRODUCTION_NOT_RESTORED: campaign service cleanup did not verify all campaign units stopped')
            return
        if not resources_free:
            log('PRODUCTION_NOT_RESTORED: GPU/ports or MemAvailable did not become free within bounded waits')
            return
        try:
            require_production_inactive('before production restore')
        except RuntimeError as e:
            log(f'PRODUCTION_NOT_RESTORED: {e}')
            return
        log('restoring production service')
        try:
            ok = start_prod()
        except (OSError, RuntimeError, json.JSONDecodeError) as e:
            log(f'production restore failed: {type(e).__name__}: {e}')
        else:
            log(f'production restored: {"ready" if ok else "NOT READY"}')

    signal.signal(signal.SIGTERM, lambda *a: (on_exit(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *a: (on_exit(), sys.exit(0)))

    try:
        if not opts.skip_ctest:
            log('ctest parity subset')
            r = sh(['ctest', '--test-dir', f'{REPO}/build-hip-port', '-R',
                    '^(pm4_test|hip_buffer_load_test|hip_moe_iu8_test|hip_moe_iu4_test|hip_projection_test|'
                    'hip_fp8_test|fp8_attn_parity|moe_iu8_parity|moe_iu4_parity|hip_projection_parity|arch_defaults|'
                    'qsa_merge_daily_parity_default|qsa_merge_daily_parity_daily|gr_upmix_parity|hip_gr_upmix_parity)$',
                    '--output-on-failure'], timeout=5400)
            with open(f'{ROOT}/results/ctest.log', 'w') as fh:
                fh.write(r.stdout + r.stderr)
            summary = [l for l in r.stdout.splitlines() if 'tests passed' in l or 'Failed' in l]
            log(f'ctest rc={r.returncode} {summary[-1] if summary else ""}')

        for rnd in range(1 + opts.round_offset, 1 + opts.round_offset + opts.rounds):
            for a in arms:
                sid = a['id']
                label = (f'{a["switch"]}: prod {a["baseline"]} -> arm {a["arm"]}' if a.get('switch')
                         else f'engine binary: {os.path.basename(os.path.dirname(ctl_exe(a))) } -> {os.path.basename(os.path.dirname(trt_exe(a)))}')
                log(f'=== PAIR {sid} ROUND {rnd} ({label}) ===')
                sfx = '' if rnd == 1 else f'_r{rnd}'
                if a.get('treatment_only'):
                    log(f'PAIR {sid} round {rnd} treatment-only '
                        '(control reused from campaign 9 ctl_relaxed_x4_dot4)')
                    ctl = None
                else:
                    ctl = run_arm(f'ctl_{sid}', suffix=sfx)
                    if ctl not in ('ok', 'route_invalid'):
                        log(f'PAIR {sid} round {rnd} ABORTED: control arm {ctl}')
                        raise RuntimeError(f'{sid} round {rnd}: control arm {ctl}; aborting campaign')
                trt = run_arm(f'trt_{sid}', suffix=sfx)
                control_label = ('reused campaign 9 ctl_relaxed_x4_dot4' if ctl is None else ctl)
                log(f'PAIR {sid} round {rnd} control={control_label} treatment={trt}')
                if trt not in ('ok', 'route_invalid'):
                    raise RuntimeError(f'{sid} round {rnd}: treatment arm {trt}; aborting campaign')
                trt_rec = json.load(open(f'{ROOT}/results/trt_{sid}{sfx}.json'))
                trt_slots = trt_rec.get('expert_cache_slots')
                expected_slots = int(a['expert_cache_slots'])
                if a.get('treatment_only'):
                    if trt_slots != expected_slots:
                        raise RuntimeError(f'{sid} round {rnd}: treatment expert-cache slots must equal '
                                           f'{expected_slots} (treatment={trt_slots})')
                else:
                    ctl_rec = json.load(open(f'{ROOT}/results/ctl_{sid}{sfx}.json'))
                    ctl_slots = ctl_rec.get('expert_cache_slots')
                    if ctl_slots != expected_slots or trt_slots != expected_slots:
                        raise RuntimeError(f'{sid} round {rnd}: expert-cache slots must both equal '
                                           f'{expected_slots} (control={ctl_slots}, treatment={trt_slots})')
                if ctl == 'route_invalid' or trt == 'route_invalid':
                    log(f'PAIR {sid} round {rnd} excluded: route evidence invalid')
                    continue
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
