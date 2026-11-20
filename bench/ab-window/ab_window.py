#!/usr/bin/env python3
"""Solo window-width A/B: STRATA_WINDOW_MAX_T=8 vs 12 vs 16 builds of merged fork-main.

Pure build isolation: same source (7782928f2), same prod env (new P1/P2/P4/P5 flags OFF),
same model/pack/args. w8 and w16 keep the prod user_args (--lookup-max-t 16); w12 is clamped
to its 12-row cap (kVerifyLongT=12). Each arm's window ladder is read from its startup banner.
Deltas reported vs w16 (the wide build). Greedy parity probe + 2 greedy decode trials + 1 sampled.

Usage:
  ab_window.py --tree <integration worktree> --plan
  ab_window.py --tree <integration worktree> --run [--launches 3]
"""
import argparse, glob, hashlib, json, os, re, shlex, shutil, statistics, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

SKILL = Path('/home/soren/.hermes/skills/software-development/strata-hip-install')
CCFG = SKILL / 'scripts/candidate_config.py'
GPUWAIT = SKILL / 'scripts/gpu_free_wait.sh'
PROD_PROFILE = Path('/home/soren/.local/share/strata-upstream-v0.2.10/components/strata-iq3_s.json')
PROD_ENV = Path('/home/soren/.config/strata-upstream-v0.2.10/production.env')
SNAP = Path('/home/soren/.hermes/cache/scratch/prod_snapshot_pre_ab.json')
OUTROOT = Path('/home/soren/.hermes/cache/scratch/ab_window')
LIVE_PACK = '/home/soren/.local/share/strata-upstream-v0.2.10/data/dflash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack'
PRODDIR = '/home/soren/.local/share/strata-upstream-v0.2.10/terminal/0.1.56'

# arm -> (port, build subdir, lookup-max-t override or None=keep prod, lookup-sizes override or None)
ARMS = {
    'w8':  {'port': 8286, 'build': 'build-hip-gfx1201-w8',      'lmax': None, 'lsizes': None},
    'w12': {'port': 8287, 'build': 'build-hip-gfx1201-w12',     'lmax': '12', 'lsizes': '10,11,12'},
    'w16': {'port': 8288, 'build': 'build-hip-gfx1201-p01245',  'lmax': None, 'lsizes': None},
}
ORDER = ['w8', 'w12', 'w16']

GREEDY_PROMPT = ("[parity-probe] Write a short Python function that returns the nth Fibonacci "
                 "number iteratively, then explain its time complexity in two sentences.")
TRIAL_PROMPTS = [
    "[solo-greedy trial {i}] Summarize the key ideas of speculative decoding in three bullet points, then give a one-line example.",
    "[solo-greedy trial {i}] Explain why a memory-bound GPU decode benefits from wider verification windows. Keep it under 120 words.",
]
SAMPLED_PROMPT = "[solo-sampled] Write a creative four-line poem about a GPU that dreams of electric sheep."

DECODE_RE = re.compile(r'(\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
REUSE_RE = re.compile(r'prompt (\d+) tokens = (\d+) reused')
ACC_RE = re.compile(r'drafts accepted (\d+) of (\d+)')


def load_base_env():
    snap = json.loads(SNAP.read_text())
    env = dict(snap['switches'])  # production effective switches; new P1/P2/P4/P5 flags absent (OFF)
    for line in PROD_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        try:
            val = shlex.split(v, comments=True)[0]
        except (ValueError, IndexError):
            val = v
        if k == 'STRATA_API_KEY':
            env[k] = val
        elif k not in env:
            env[k] = val
    env.update({'LD_LIBRARY_PATH': '/opt/rocm/lib',
                'PATH': '/home/soren/Strata/.venv/bin:/home/soren/.local/bin:/usr/sbin:/usr/bin:/bin:/opt/rocm/bin',
                'PYTHONUNBUFFERED': '1', 'ROCM_PATH': '/opt/rocm', 'HIP_PATH': '/opt/rocm'})
    return env, snap


def arm_env(base, arm):
    e = dict(base)
    dump = OUTROOT / arm / 'accept_dump'; dump.mkdir(parents=True, exist_ok=True)
    e['STRATA_DF2_CAND_DUMP'] = str(dump)
    return e


def write_env_file(e, path):
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
    # per-arm lookup window override in user_args (w12 clamped to its 12-row cap)
    ua = list(cfg.get('user_args', []))
    if spec['lmax'] is not None:
        for i, x in enumerate(ua):
            if x == '--lookup-max-t':
                ua[i + 1] = spec['lmax']
            elif x == '--lookup-sizes' and spec['lsizes'] is not None:
                ua[i + 1] = spec['lsizes']
        cfg['user_args'] = ua
    out.write_text(json.dumps(cfg, indent=1)); os.chmod(out, 0o600)
    return out, cfg


def http_json(url, key=None, body=None, timeout=600):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['x-api-key'] = key
        headers['Authorization'] = '***' + key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method='POST' if data else 'GET')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


def wait_ready(port, key, deadline=240):
    t0 = time.time(); last = None
    while time.time() - t0 < deadline:
        st, _ = http_json(f'http://127.0.0.1:{port}/v1/models', key=key, timeout=10)
        if st == 200:
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10); unkeyed = 200
            except urllib.error.HTTPError as e:
                unkeyed = e.code
            except Exception:
                unkeyed = None
            return True, unkeyed
        last = st
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
    for d in glob.glob('/proc/[0-9]*'):
        try:
            if os.path.realpath(os.readlink(d + '/exe')) == target:
                return int(os.path.basename(d))
        except OSError:
            pass
    return None


def chat(port, key, model, prompt, max_tokens, temp):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': temp, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}
    st, o = http_json(f'http://127.0.0.1:{port}/v1/chat/completions', key=key, body=body)
    if not o:
        return {'status': st, 'error': 'no body'}
    ch = o['choices'][0]; content = ch.get('message', {}).get('content', '') or ''
    u = o.get('usage', {})
    return {'status': st, 'finish': ch.get('finish_reason'), 'content': content,
            'sha256': hashlib.sha256(content.encode()).hexdigest(),
            'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': u.get('completion_tokens')}


def parse_new_log(logpath, offset):
    txt = Path(logpath).read_text(errors='replace') if Path(logpath).exists() else ''
    new = txt[offset:]
    dec = [(int(m.group(1)), int(m.group(2)), float(m.group(3))) for m in DECODE_RE.finditer(new)]
    acc = [(int(m.group(1)), int(m.group(2))) for m in ACC_RE.finditer(new)]
    return dec, acc, len(txt)


def launch(arm, exe, cfgpath, envfile, env, idx):
    spec = ARMS[arm]; port = spec['port']; key = env['STRATA_API_KEY']
    unit = f'ab-win-{arm}-{idx}'; scratch = OUTROOT / arm
    stdout = scratch / f'stdout-{idx}.log'; logpath = scratch / 'engine.log'
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'strata.service']).returncode == 0:
        raise RuntimeError('production active; abort')
    g = subprocess.run(['bash', str(GPUWAIT), str(port), '300'], capture_output=True, text=True)
    if g.returncode:
        raise RuntimeError(f'gpu free fail: {g.stdout[-200:]}')
    avail = int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))) // 1048576
    if avail < 52:
        raise RuntimeError(f'MemAvailable {avail} < 52')
    if Path(logpath).exists():
        Path(logpath).unlink()
    cmd = (f"set -a; . {envfile}; set +a; cd {PRODDIR} && "
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
            rec['stdout_tail'] = Path(stdout).read_text(errors='replace')[-1200:] if Path(stdout).exists() else ''
            return rec
        time.sleep(1)
        banner = Path(logpath).read_text(errors='replace') if Path(logpath).exists() else ''
        rec['df2_banner'] = 'DFlash2 drafter on' in banner
        est = [l for l in banner.splitlines() if 'engine started:' in l]
        if est:
            lav = shlex.split(est[-1].split('engine started:', 1)[1])
            rec['log_exe_matches'] = bool(lav) and os.path.realpath(lav[0]) == os.path.realpath(str(exe))
            dm = [lav[i + 1] for i, x in enumerate(lav) if x == '--draft-model']
            rec['pack_matches_live'] = (dm == [LIVE_PACK])
            rec['log_solo'] = '--batch' not in lav
            rec['log_lookup_max_t'] = [lav[i + 1] for i, x in enumerate(lav) if x == '--lookup-max-t']
            rec['log_lookup_sizes'] = [lav[i + 1] for i, x in enumerate(lav) if x == '--lookup-sizes']
            if not rec['pack_matches_live'] or not rec['log_exe_matches']:
                rec['error'] = f'pack/exe mismatch: dm={dm} exe_ok={rec["log_exe_matches"]}'
                rec['ok'] = False
                return rec
        # window ladder banner
        lad = [l for l in banner.splitlines() if 'window sizes' in l or 'windows up to' in l or 'window warm-up' in l]
        rec['window_ladder'] = [l[:200] for l in lad[-3:]]
        mpid = int(subprocess.run(['systemctl', '--user', 'show', f'{unit}.service', '-p', 'MainPID', '--value'],
                                  capture_output=True, text=True).stdout.strip() or 0)
        epid = find_engine(mpid, exe)
        if epid:
            eenv = dict(x.split('=', 1) for x in Path(f'/proc/{epid}/environ').read_text(errors='replace').split('\0') if '=' in x)
            ce = subprocess.run([sys.executable, str(SKILL / 'scripts/check_engine_env.py'), str(exe), str(envfile)],
                                capture_output=True, text=True)
            rec['check_engine_env_rc'] = ce.returncode
            rec['check_engine_env_out'] = ce.stdout.strip()[-300:]
        _, models = http_json(f'http://127.0.0.1:{port}/v1/models', key=key)
        model = models['data'][0]['id'] if models else None
        rec['model'] = model
        chat(port, key, model, 'Warmup: reply with the single word OK.', 16, 0.0)
        off = len(Path(logpath).read_text(errors='replace')) if Path(logpath).exists() else 0
        probe = chat(port, key, model, GREEDY_PROMPT, 256, 0.0)
        dec, acc, off = parse_new_log(logpath, off); probe['decode'] = dec; probe['acc'] = acc
        rec['parity_probe'] = probe
        rec['greedy_trials'] = []
        for i, tpl in enumerate(TRIAL_PROMPTS, 1):
            off = len(Path(logpath).read_text(errors='replace')) if Path(logpath).exists() else 0
            r = chat(port, key, model, tpl.format(i=i), 512, 0.0)
            dec, acc, off = parse_new_log(logpath, off); r['decode'] = dec; r['acc'] = acc
            rec['greedy_trials'].append(r)
        off = len(Path(logpath).read_text(errors='replace')) if Path(logpath).exists() else 0
        s = chat(port, key, model, SAMPLED_PROMPT, 512, 0.7)
        dec, acc, off = parse_new_log(logpath, off); s['decode'] = dec; s['acc'] = acc
        rec['sampled_trial'] = s
        rec['ok'] = True
    finally:
        subprocess.run(['systemctl', '--user', 'stop', f'{unit}.service'], capture_output=True, text=True)
        subprocess.run(['bash', str(GPUWAIT), str(port), '180'], capture_output=True, text=True)
    return rec


def aggregate(records):
    summ = {}
    for arm in ORDER:
        rs = [r for r in records if r['arm'] == arm and r.get('ok')]
        def rates(cell):
            vals = []
            for r in rs:
                items = r.get(cell) if isinstance(r.get(cell), list) else [r.get(cell)]
                for it in items:
                    if it and it.get('decode'):
                        vals.append(it['decode'][-1][2])
            return vals
        def acc_ratio(cell):
            a = b = 0
            for r in rs:
                items = r.get(cell) if isinstance(r.get(cell), list) else [r.get(cell)]
                for it in items:
                    for x in (it.get('acc') or []):
                        a += x[0]; b += x[1]
            return round(a / b, 3) if b else None
        g = rates('greedy_trials'); s = rates('sampled_trial')
        summ[arm] = {
            'n_launches': len(rs),
            'greedy_decode_tok_s': {'n': len(g), 'mean': round(statistics.mean(g), 1) if g else None,
                                    'min': round(min(g), 1) if g else None, 'max': round(max(g), 1) if g else None},
            'sampled_decode_tok_s': {'n': len(s), 'mean': round(statistics.mean(s), 1) if s else None,
                                     'min': round(min(s), 1) if s else None, 'max': round(max(s), 1) if s else None},
            'greedy_accept_ratio': acc_ratio('greedy_trials'),
            'parity_sha256': sorted({r['parity_probe']['sha256'] for r in rs if r.get('parity_probe')}),
            'window_ladder': rs[-1].get('window_ladder') if rs else None,
            'lookup_max_t': rs[-1].get('log_lookup_max_t') if rs else None,
        }
    ref = summ['w16']['greedy_decode_tok_s']['mean']
    for arm in ORDER:
        m = summ[arm]['greedy_decode_tok_s']['mean']
        summ[arm]['greedy_delta_vs_w16_pct'] = round((m - ref) / ref * 100, 1) if m and ref else None
    refs = summ['w16']['sampled_decode_tok_s']['mean']
    for arm in ORDER:
        m = summ[arm]['sampled_decode_tok_s']['mean']
        summ[arm]['sampled_delta_vs_w16_pct'] = round((m - refs) / refs * 100, 1) if m and refs else None
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--launches', type=int, default=3)
    a = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    base, snap = load_base_env()
    exes = {arm: str(Path(a.tree) / ARMS[arm]['build'] / 'strata') for arm in ORDER}
    cfgs = {}; envfiles = {}; envs = {}
    for arm in ORDER:
        cfg, _ = gen_arm_config(arm, exes[arm], a.tree)
        e = arm_env(base, arm); ef = OUTROOT / f'{arm}.env'; write_env_file(e, ef)
        cfgs[arm] = cfg; envfiles[arm] = ef; envs[arm] = e
    if a.plan:
        print('=== PLAN (window-width A/B) ===')
        for arm in ORDER:
            cfg = json.loads(cfgs[arm].read_text())
            print(f'\n[{arm}] exe={exes[arm]} exists={Path(exes[arm]).exists()}')
            if Path(exes[arm]).exists():
                print('  sha256', hashlib.sha256(Path(exes[arm]).read_bytes()).hexdigest()[:16])
            print('  port', cfg['port'], 'parallel', cfg['parallel'])
            print('  user_args', cfg.get('user_args'))
        # env identical across arms except CAND_DUMP
        diffs = {k for k in set(envs['w8']) | set(envs['w16']) if envs['w8'].get(k) != envs['w16'].get(k)}
        print('\nenv_diff_w8_vs_w16', sorted(diffs))
        print('new_flags_present', {k: envs['w16'].get(k) for k in ('STRATA_DF2_TAP_OVERLAP_DECODE','STRATA_DF2_ADAPTIVE_SOLO','STRATA_DF2_SUFFIX_FILL','STRATA_DF2_SAMPLE_WALK')})
        print('PLAN_OK')
        return
    if a.run:
        records = []
        for i in range(1, a.launches + 1):
            for arm in ORDER:
                print(f'=== LAUNCH {arm} #{i} ===', flush=True)
                try:
                    rec = launch(arm, exes[arm], cfgs[arm], envfiles[arm], envs[arm], i)
                except Exception as e:
                    rec = {'arm': arm, 'idx': i, 'ok': False, 'error': str(e)}
                records.append(rec)
                (OUTROOT / 'records.jsonl').open('a').write(json.dumps(rec) + '\n')
                print(json.dumps({k: rec.get(k) for k in ('arm','idx','ok','ready','pack_matches_live','log_exe_matches','check_engine_env_rc','error')}), flush=True)
                if not rec.get('ok'):
                    print(f'ABORT after failed launch {arm} #{i}: {rec.get("error")}', flush=True)
                    break
            else:
                continue
            break
        summ = aggregate(records)
        (OUTROOT / 'summary.json').write_text(json.dumps(summ, indent=1))
        print('=== SUMMARY ==='); print(json.dumps(summ, indent=1))
        doc = Path(a.tree) / 'bench' / 'ab-window'; doc.mkdir(parents=True, exist_ok=True)
        for f in ['records.jsonl', 'summary.json']:
            if (OUTROOT / f).exists():
                shutil.copy2(OUTROOT / f, doc / f)
        shutil.copy2(Path(__file__), doc / 'ab_window.py')
        for arm in ORDER:
            for f in [f'{arm}.json', f'{arm}.env']:
                if (OUTROOT / f).exists():
                    shutil.copy2(OUTROOT / f, doc / f)
        print('DOC_COPIED', doc)


if __name__ == '__main__':
    main()
