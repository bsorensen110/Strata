#!/usr/bin/env python3
"""Engine-only solo A/B: current production binary (baseline) vs merged P0/P1/P2/P4/P5 build (treatment).

Both arms run from the integration worktree's serve layer (matched python), parallel=1,
same model/pack/args. Baseline env = captured production effective switches. Treatment adds
the four new opt-in flags. Greedy parity probe + greedy decode trials + one sampled trial
(exercises P5, which is inactive under greedy).

Usage:
  ab_p01245.py --treatment-exe <path> --tree <integration worktree> --plan
  ab_p01245.py --treatment-exe <path> --tree <integration worktree> --run [--launches 3]
"""
import argparse, glob, hashlib, json, os, re, shlex, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

SKILL = Path('/home/soren/.hermes/skills/software-development/strata-hip-install')
CCFG = SKILL / 'scripts/candidate_config.py'
GPUWAIT = SKILL / 'scripts/gpu_free_wait.sh'
PROD_PROFILE = Path('/home/soren/.local/share/strata-upstream-v0.2.10/components/strata-iq3_s.json')
PROD_ENV = Path('/home/soren/.config/strata-upstream-v0.2.10/production.env')
SNAP = Path('/home/soren/.hermes/cache/scratch/prod_snapshot_pre_ab.json')
PY = '/home/soren/Strata/.venv/bin/python'
OUTROOT = Path('/home/soren/.hermes/cache/scratch/ab_p01245')

NEW_FLAGS = {
    'STRATA_DF2_TAP_OVERLAP_DECODE': '1',   # P1
    'STRATA_DF2_ADAPTIVE_SOLO': '1',        # P2
    'STRATA_DF2_SUFFIX_FILL': '1',          # P4
    'STRATA_DF2_SAMPLE_WALK': '1',          # P5 (non-greedy T>0 only)
}
ARMS = {'baseline': {'port': 8286, 'new': False},
        'treatment': {'port': 8287, 'new': True}}
LIVE_PACK = '/home/soren/.local/share/strata-upstream-v0.2.10/data/dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack'

GREEDY_PROMPT = ("[parity-probe] Write a short Python function that returns the nth Fibonacci "
                 "number iteratively, then explain its time complexity in two sentences.")
TRIAL_PROMPTS = [
    "[solo-greedy trial {i}] Summarize the key ideas of speculative decoding in three bullet points, then give a one-line example.",
    "[solo-greedy trial {i}] Explain why a memory-bound GPU decode benefits from wider verification windows. Keep it under 120 words.",
]
SAMPLED_PROMPT = "[solo-sampled] Write a creative four-line poem about a GPU that dreams of electric sheep."


def load_base_env():
    snap = json.loads(SNAP.read_text())
    env = dict(snap['switches'])  # effective production STRATA_* switches (incl arch defaults + paths)
    # API key + any non-STRATA vars from production.env (never print key)
    for line in PROD_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        try:
            parts = shlex.split(v, comments=True)
        except ValueError:
            parts = [v]
        val = parts[0] if parts else ''
        if k == 'STRATA_API_KEY':
            env[k] = val
        elif k not in env:
            env[k] = val
    env['LD_LIBRARY_PATH'] = '/opt/rocm/lib'
    env['PATH'] = '/home/soren/Strata/.venv/bin:/home/soren/.local/bin:/usr/sbin:/usr/bin:/bin:/opt/rocm/bin'
    env['PYTHONUNBUFFERED'] = '1'
    env['ROCM_PATH'] = '/opt/rocm'
    env['HIP_PATH'] = '/opt/rocm'
    return env, snap


def arm_env(base, arm):
    e = dict(base)
    # isolate the P0 candidate-dump output per arm (prod points at a shared dir)
    dump = OUTROOT / arm / 'accept_dump'
    dump.mkdir(parents=True, exist_ok=True)
    e['STRATA_DF2_CAND_DUMP'] = str(dump)
    if ARMS[arm]['new']:
        e.update(NEW_FLAGS)
    return e


def write_env_file(e, path):
    # 'export ' prefix so check_engine_env.py parses it; still valid when sourced.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as fh:
        for k in sorted(e):
            fh.write(f'export {k}={shlex.quote(e[k])}\n')
    os.chmod(path, 0o600)


def gen_arm_config(arm, exe, tree):
    spec = ARMS[arm]
    scratch = OUTROOT / arm
    (scratch / 'conv-cache').mkdir(parents=True, exist_ok=True)
    (scratch / 'slots').mkdir(parents=True, exist_ok=True)
    out = OUTROOT / f'{arm}.json'
    cmd = [sys.executable, str(CCFG), str(PROD_PROFILE), str(out), '--exe', str(exe),
           '--port', str(spec['port']), '--scratch', str(scratch),
           f'--set=--expert-profile-save={scratch}/learned.bin',
           f'--set=--conversation-cache-dir={scratch}/conv-cache']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'candidate_config {arm}: {r.stdout} {r.stderr}')
    cfg = json.loads(out.read_text())
    cfg['parallel'] = 1
    out.write_text(json.dumps(cfg, indent=1)); os.chmod(out, 0o600)
    return out, cfg


def http_json(url, key=None, body=None, timeout=600):
    headers = {'Content-Type': 'application/json'}
    if key:
        # this serve build authenticates via x-api-key; Bearer alone returned 401.
        headers['x-api-key'] = key
        headers['Authorization'] = '***' + key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method='POST' if data else 'GET')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def wait_ready(port, key, deadline=240):
    t0 = time.time()
    last = None
    while time.time() - t0 < deadline:
        try:
            st, _ = http_json(f'http://127.0.0.1:{port}/v1/models', key=key, timeout=10)
            if st == 200:
                # unkeyed must be 401
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10)
                    unkeyed = 200
                except urllib.error.HTTPError as e:
                    unkeyed = e.code
                return True, unkeyed
            last = st
        except Exception as e:
            last = str(e)[:80]
        time.sleep(3)
    return False, last


def find_engine(mainpid, exe):
    target = os.path.realpath(str(exe))
    try:
        kids = [int(x) for x in Path(f'/proc/{mainpid}/task/{mainpid}/children').read_text().split()]
    except Exception:
        kids = []
    for k in kids:
        try:
            if os.path.realpath(os.readlink(f'/proc/{k}/exe')) == target:
                return k
        except Exception:
            pass
    # the engine may not be a direct child of the serve MainPID; scan by real binary path
    for d in glob.glob('/proc/[0-9]*'):
        try:
            if os.path.realpath(os.readlink(d + '/exe')) == target:
                return int(os.path.basename(d))
        except OSError:
            pass
    return None


def engine_facts(pid):
    av = [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    env = dict(x.split('=', 1) for x in Path(f'/proc/{pid}/environ').read_text(errors='replace').split('\0') if '=' in x)
    return av, env


def chat(port, key, model, prompt, max_tokens, temp, marker=None):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': temp, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}
    st, o = http_json(f'http://127.0.0.1:{port}/v1/chat/completions', key=key, body=body)
    ch = o['choices'][0]
    content = ch.get('message', {}).get('content', '') or ''
    usage = o.get('usage', {})
    return {'status': st, 'finish': ch.get('finish_reason'), 'content': content,
            'sha256': hashlib.sha256(content.encode()).hexdigest(),
            'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens')}


DECODE_RE = re.compile(r'(\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
REUSE_RE = re.compile(r'prompt (\d+) tokens = (\d+) reused')
ACC_RE = re.compile(r'drafts accepted (\d+) of (\d+)')


def parse_new_log(logpath, offset):
    txt = Path(logpath).read_text(errors='replace') if Path(logpath).exists() else ''
    new = txt[offset:]
    dec = [(int(m.group(1)), int(m.group(2)), float(m.group(3))) for m in DECODE_RE.finditer(new)]
    reuse = [(int(m.group(1)), int(m.group(2))) for m in REUSE_RE.finditer(new)]
    acc = [(int(m.group(1)), int(m.group(2))) for m in ACC_RE.finditer(new)]
    return dec, reuse, acc, len(txt)


def launch(arm, exe, tree, cfgpath, envfile, env, idx):
    spec = ARMS[arm]
    port = spec['port']
    key = env['STRATA_API_KEY']
    unit = f'ab-p01245-{arm}-{idx}'
    scratch = OUTROOT / arm
    stdout = scratch / f'stdout-{idx}.log'
    logpath = scratch / 'engine.log'
    # gates
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'strata.service']).returncode == 0:
        raise RuntimeError('production service active; abort')
    g = subprocess.run(['bash', str(GPUWAIT), str(port), '300'], capture_output=True, text=True)
    if g.returncode:
        raise RuntimeError(f'gpu free fail: {g.stdout[-300:]}')
    avail = int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))) // 1048576
    if avail < 52:
        raise RuntimeError(f'MemAvailable {avail} GiB < 52')
    # clear per-launch engine log to avoid accumulation (fresh launch)
    if Path(logpath).exists():
        Path(logpath).unlink()
    proddir = '/home/soren/.local/share/strata-upstream-v0.2.10/terminal/0.1.56'
    cmd = (f"set -a; . {envfile}; set +a; cd {proddir} && "
           f"exec /home/soren/.local/bin/strata serve --profile {cfgpath} --output jsonl > {stdout} 2>&1")
    subprocess.run(['systemd-run', '--user', f'--unit={unit}', '--collect', '--no-block',
                    '-p', 'MemoryMax=infinity', '-p', 'MemorySwapMax=infinity', '-p', 'TasksMax=infinity',
                    '--', '/bin/bash', '-c', cmd], check=True, capture_output=True, text=True)
    rec = {'arm': arm, 'idx': idx, 'unit': unit, 'port': port, 'exe': str(exe)}
    try:
        ok, unkeyed = wait_ready(port, key)
        rec['ready'] = ok; rec['unkeyed_status'] = unkeyed
        if not ok:
            rec['error'] = f'not ready: {unkeyed}'
            rec['stdout_tail'] = Path(stdout).read_text(errors='replace')[-1500:] if Path(stdout).exists() else ''
            return rec
        mpid = int(subprocess.run(['systemctl', '--user', 'show', f'{unit}.service', '-p', 'MainPID', '--value'],
                                  capture_output=True, text=True).stdout.strip())
        epid = find_engine(mpid, exe)
        rec['engine_pid'] = epid
        if epid:
            av, eenv = engine_facts(epid)
            rec['argv_ok'] = ('--serve' in av and '--draft' in av and 'dflash2' in av
                              and any('q4_0-selq4' in x for x in av))
            dm = [av[i + 1] for i, x in enumerate(av) if x == '--draft-model']
            rec['live_pack'] = dm
            rec['pack_matches_live'] = (dm == [LIVE_PACK])
            if not rec['pack_matches_live']:
                rec['error'] = f'live draft pack {dm} != production {LIVE_PACK}'
                rec['ok'] = False
                return rec
            bad = [k for k, v in env.items() if k.startswith('STRATA_') and k != 'STRATA_API_KEY' and eenv.get(k) != v]
            rec['env_mismatch'] = bad
            rec['new_flags_live'] = {k: eenv.get(k) for k in NEW_FLAGS} if spec['new'] else {}
            ce = subprocess.run([sys.executable, str(SKILL / 'scripts/check_engine_env.py'), str(exe), str(envfile)],
                                capture_output=True, text=True)
            rec['check_engine_env_rc'] = ce.returncode
            rec['check_engine_env_out'] = ce.stdout.strip()[-400:]
        # model id
        _, models = http_json(f'http://127.0.0.1:{port}/v1/models', key=key)
        model = models['data'][0]['id']
        rec['model'] = model
        # banner
        time.sleep(1)
        banner = Path(logpath).read_text(errors='replace') if Path(logpath).exists() else ''
        rec['df2_banner'] = 'DFlash2 drafter on' in banner
        # authoritative argv from the engine.log 'engine started:' line (survives process exit)
        est = [l for l in banner.splitlines() if 'engine started:' in l]
        if est:
            lav = shlex.split(est[-1].split('engine started:', 1)[1])
            rec['log_exe'] = lav[0] if lav else None
            rec['log_exe_matches'] = bool(lav) and os.path.realpath(lav[0]) == os.path.realpath(str(exe))
            ldm = [lav[i + 1] for i, x in enumerate(lav) if x == '--draft-model']
            rec['log_pack'] = ldm
            rec['log_pack_matches_live'] = (ldm == [LIVE_PACK])
            rec['log_solo'] = '--batch' not in lav
            rec['log_spec'] = [lav[i + 1] for i, x in enumerate(lav) if x == '--spec']
            rec['log_draft_conf'] = [lav[i + 1] for i, x in enumerate(lav) if x == '--draft-conf']
        # warmup (not timed)
        chat(port, key, model, 'Warmup: reply with the single word OK.', 16, 0.0)
        # greedy parity probe
        off = len(Path(logpath).read_text(errors='replace')) if Path(logpath).exists() else 0
        probe = chat(port, key, model, GREEDY_PROMPT, 256, 0.0)
        dec, reuse, acc, off = parse_new_log(logpath, off)
        probe['decode'] = dec; probe['acc'] = acc
        rec['parity_probe'] = probe
        # greedy timed trials
        rec['greedy_trials'] = []
        for i, tpl in enumerate(TRIAL_PROMPTS, 1):
            off = len(Path(logpath).read_text(errors='replace')) if Path(logpath).exists() else 0
            r = chat(port, key, model, tpl.format(i=i), 512, 0.0)
            dec, reuse, acc, off = parse_new_log(logpath, off)
            r['decode'] = dec; r['reuse'] = reuse; r['acc'] = acc
            rec['greedy_trials'].append(r)
        # sampled trial (exercises P5)
        off = len(Path(logpath).read_text(errors='replace')) if Path(logpath).exists() else 0
        s = chat(port, key, model, SAMPLED_PROMPT, 512, 0.7)
        dec, reuse, acc, off = parse_new_log(logpath, off)
        s['decode'] = dec; s['acc'] = acc
        rec['sampled_trial'] = s
        # feature dispatch evidence from the full engine log (env-set != engaged)
        full = Path(logpath).read_text(errors='replace') if Path(logpath).exists() else ''
        feat = {}
        for label, pat in [('accept_hist', r'accept[_ ]hist|position reach|accepts by position|reach='),
                           ('adaptive', r'adaptive|tier'), ('suffix', r'suffix'),
                           ('sample_walk', r'sample[_ ]walk|sampled walk'), ('lookup', r'lookup')]:
            ms = [l for l in full.splitlines() if re.search(pat, l, re.I)]
            feat[label] = {'count': len(ms), 'sample': [m[:200] for m in ms[-3:]]}
        rec['feature_telemetry'] = feat
        rec['ok'] = True
    finally:
        subprocess.run(['systemctl', '--user', 'stop', f'{unit}.service'], capture_output=True, text=True)
        subprocess.run(['bash', str(GPUWAIT), str(port), '180'], capture_output=True, text=True)
    return rec


def aggregate(records):
    import statistics
    summ = {}
    for arm in ARMS:
        rs = [r for r in records if r['arm'] == arm and r.get('ok')]
        def rates(cell):
            vals = []
            for r in rs:
                items = r.get(cell) if isinstance(r.get(cell), list) else [r.get(cell)]
                for it in items:
                    if it and it.get('decode'):
                        # last decode line = the request's generated rate
                        vals.append(it['decode'][-1][2])
            return vals
        g = rates('greedy_trials'); s = rates('sampled_trial')
        summ[arm] = {
            'n_launches': len(rs),
            'greedy_decode_tok_s': {'n': len(g), 'mean': round(statistics.mean(g), 1) if g else None,
                                    'min': round(min(g), 1) if g else None, 'max': round(max(g), 1) if g else None},
            'sampled_decode_tok_s': {'n': len(s), 'mean': round(statistics.mean(s), 1) if s else None,
                                     'min': round(min(s), 1) if s else None, 'max': round(max(s), 1) if s else None},
            'parity_sha256': sorted({r['parity_probe']['sha256'] for r in rs if r.get('parity_probe')}),
        }
    b = summ['baseline']['greedy_decode_tok_s']['mean']; t = summ['treatment']['greedy_decode_tok_s']['mean']
    summ['greedy_delta_pct'] = round((t - b) / b * 100, 1) if b and t else None
    bs = summ['baseline']['sampled_decode_tok_s']['mean']; ts = summ['treatment']['sampled_decode_tok_s']['mean']
    summ['sampled_delta_pct'] = round((ts - bs) / bs * 100, 1) if bs and ts else None
    summ['greedy_parity_match'] = (summ['baseline']['parity_sha256'] == summ['treatment']['parity_sha256']
                                   and len(summ['baseline']['parity_sha256']) == 1)
    return summ


def plan_check(arm, cfg, envfile):
    """serve --plan preflight under the arm env; verify live selq4 pack + solo argv."""
    cmd = f"set -a; . {envfile}; set +a; /home/soren/.local/bin/strata serve --profile {cfg} --plan --output json"
    r = subprocess.run(['/bin/bash', '-c', cmd], capture_output=True, text=True, timeout=120)
    out = r.stdout
    try:
        j = json.loads(out)
    except Exception:
        a, b = out.find('{'), out.rfind('}')
        j = json.loads(out[a:b + 1]) if a >= 0 and b > a else {}
    plan = j.get('plan', j)
    av = plan.get('engine_argv', [])
    dm = [av[i + 1] for i, x in enumerate(av) if x == '--draft-model']
    return {
        'rc': r.returncode,
        'draft_model': dm,
        'pack_matches_live': dm == [LIVE_PACK],
        'draft_dflash2': '--draft' in av and 'dflash2' in av,
        'spec': [av[i + 1] for i, x in enumerate(av) if x == '--spec'],
        'draft_conf': [av[i + 1] for i, x in enumerate(av) if x == '--draft-conf'],
        'batch_present': '--batch' in av,
        'notes': plan.get('notes', []),
        'stderr_tail': r.stderr[-300:],
    }


def parser_selftest():
    lines = [
        'prompt 222006 tokens = 0 reused + 222006 read in 62841 ms (3532.8 tok/s), 1800 generated in 23512 ms (76.6 tok/s), drafts accepted 1147 of 2229, 6 checkpoints',
        'prompt 4096 tokens = 3900 reused + 196 read in 50 ms (3920.0 tok/s), 512 generated in 6000 ms (85.3 tok/s), drafts accepted 400 of 512',
    ]
    ok = True
    for l in lines:
        d = DECODE_RE.search(l); r = REUSE_RE.search(l); a = ACC_RE.search(l)
        if not (d and r and a):
            ok = False
        else:
            assert float(d.group(3)) > 0
    # absent telemetry must stay unavailable, not invented
    assert DECODE_RE.search('no telemetry here') is None
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--treatment-exe', required=True)
    ap.add_argument('--tree', required=True)
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--launches', type=int, default=3)
    a = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    base, snap = load_base_env()
    baseline_exe = snap['engine_exe']
    exes = {'baseline': baseline_exe, 'treatment': a.treatment_exe}
    cfgs = {}; envfiles = {}; envs = {}
    for arm in ARMS:
        cfg, _ = gen_arm_config(arm, exes[arm], a.tree)
        e = arm_env(base, arm)
        ef = OUTROOT / f'{arm}.env'; write_env_file(e, ef)
        cfgs[arm] = cfg; envfiles[arm] = ef; envs[arm] = e
    if a.plan:
        print('=== PLAN ===')
        print('baseline_exe', baseline_exe, 'sha256', snap['engine_sha256'][:16])
        print('treatment_exe', a.treatment_exe, 'exists', Path(a.treatment_exe).exists())
        if Path(a.treatment_exe).exists():
            print('treatment_sha256', hashlib.sha256(Path(a.treatment_exe).read_bytes()).hexdigest()[:16])
        for arm in ARMS:
            cfg = json.loads(cfgs[arm].read_text())
            print(f'\n[{arm}] port={cfg["port"]} parallel={cfg["parallel"]} exe={cfg["exe"]}')
            print('  log', cfg['log']); print('  conv-cache-dir', [cfg['args'][i+1] for i,x in enumerate(cfg['args']) if x=='--conversation-cache-dir'])
            print('  new_flags', {k: envs[arm].get(k) for k in NEW_FLAGS})
            print('  switch_count', sum(1 for k in envs[arm] if k.startswith('STRATA_')))
        # plan diff: treatment vs baseline env
        diff = {k for k in set(envs['baseline']) | set(envs['treatment']) if envs['baseline'].get(k) != envs['treatment'].get(k)}
        print('\nenv_diff_baseline_vs_treatment', sorted(diff))
        print('\n=== profile_diff baseline vs treatment ===')
        pd = subprocess.run([sys.executable, str(SKILL / 'scripts/profile_diff.py'), str(cfgs['baseline']), str(cfgs['treatment'])],
                            capture_output=True, text=True)
        print(pd.stdout.strip()[:1200])
        print('\n=== serve --plan preflight (solo argv; pack verified LIVE at readiness) ===')
        plan_ok = True
        for arm in ARMS:
            pc = plan_check(arm, cfgs[arm], envfiles[arm])
            print(f'[{arm}]', json.dumps(pc))
            # --plan passes engine=None so it names the stock pack; the real launch
            # resolves the converted selq4 pack (both exes carry the marker). The
            # authoritative pack check is the live /proc cmdline at readiness.
            if not (pc['draft_dflash2'] and not pc['batch_present']
                    and pc['spec'] == ['8'] and pc['draft_conf'] == ['0.35']):
                plan_ok = False
        print('note: plan pack is engine=None artifact; live pack hard-checked per launch')
        print('\nparser_selftest', parser_selftest())
        print('PLAN_PREFLIGHT_OK' if plan_ok else 'PLAN_PREFLIGHT_FAIL')
        print('PLAN_OK')
        return
    if a.run:
        records = []
        schedule = []
        for i in range(1, a.launches + 1):
            schedule += [('baseline', i), ('treatment', i)]
        for arm, i in schedule:
            print(f'=== LAUNCH {arm} #{i} ===', flush=True)
            try:
                rec = launch(arm, exes[arm], a.tree, cfgs[arm], envfiles[arm], envs[arm], i)
            except Exception as e:
                rec = {'arm': arm, 'idx': i, 'ok': False, 'error': str(e)}
            records.append(rec)
            (OUTROOT / 'records.jsonl').open('a').write(json.dumps(rec) + '\n')
            print(json.dumps({k: rec.get(k) for k in ('arm','idx','ok','ready','df2_banner','argv_ok','env_mismatch','error')}), flush=True)
            if not rec.get('ok'):
                print(f'ABORT after failed launch {arm} #{i}: {rec.get("error")}', flush=True)
                break
        summ = aggregate(records)
        (OUTROOT / 'summary.json').write_text(json.dumps(summ, indent=1))
        print('=== SUMMARY ==='); print(json.dumps(summ, indent=1))
        print('RESULTS', OUTROOT / 'records.jsonl')
        # copy durable evidence into the repo worktree (scratch is pruned after 24h)
        doc = Path(a.tree) / 'bench' / 'ab-p01245'
        doc.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in ['records.jsonl', 'summary.json']:
            if (OUTROOT / f).exists():
                shutil.copy2(OUTROOT / f, doc / f)
        shutil.copy2(Path(__file__), doc / 'ab_p01245.py')
        for arm in ARMS:
            for f in [f'{arm}.json', f'{arm}.env']:
                if (OUTROOT / f).exists():
                    shutil.copy2(OUTROOT / f, doc / f)
        print('DOC_COPIED', doc)


if __name__ == '__main__':
    main()
