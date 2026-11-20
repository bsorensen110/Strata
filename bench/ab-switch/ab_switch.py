#!/usr/bin/env python3
"""Per-switch solo ablation of the Nov-19 verify-wide-all optimization set (deep-context).

Isolates ENV switches on the SAME prod binary (fork-58aefafc9-p0). Control = current prod env
(3 found culprits already reverted to 0). Each ablation arm reverts ONE solo-active suspect to
its pre-promotion value (diagnosis docs/ab/decode-dropout-diagnosis-2026-11-19.md §5). The 7
batch-only/no-op switches are excluded (inert at parallel=1). Deep-context workload (~222k prompt,
max_tokens 1024, T0) is required: wide-window switches only fire on deep lookup-chain windows.

Usage:
  ab_switch.py --plan
  ab_switch.py --run [--launches 3] [--arms control,draft_tail_skip,...]
"""
import argparse, json, os, shlex, shutil, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, '/home/soren/.hermes/cache/scratch')
import ab_window as W  # reuse http_json/wait_ready/find_engine/chat/parse_new_log/load_base_env/write_env_file

SKILL = W.SKILL
CCFG = W.CCFG
GPUWAIT = W.GPUWAIT
PROD_PROFILE = W.PROD_PROFILE
OUTROOT = Path('/home/soren/.hermes/cache/scratch/ab_switch')
DEEP_REQ = Path('/home/soren/.hermes/cache/scratch/p0_deep_request.json')
PROD_EXE = '/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-58aefafc9-p0/strata'
LIVE_PACK = W.LIVE_PACK

# arm -> {switch: pre-promotion revert value}. control = no change.
# Solo-active suspects only (diagnosis §5); 7 batch-only/no-op excluded.
ABLATIONS = {
    'control':            {},
    'bucket_nosplit':     {'STRATA_MOE_BUCKET_NOSPLIT': '0', 'STRATA_MOE_BUCKET_CHUNK': '4'},
    'draft_tail_skip':    {'STRATA_DRAFT_TAIL_SKIP': '0'},
    'pcie_fetch_overlap': {'STRATA_PCIE_FETCH_OVERLAP': '0'},
    'staging_blobs':      {'STRATA_STAGING_BLOBS': '16'},
    'pcie_group_rows':    {'STRATA_PCIE_GROUP_ROWS': '16'},
    'cpu_reduce_miss':    {'STRATA_CPU_REDUCE_MISS': '0'},
    'cpu_expert_no_split':{'STRATA_CPU_EXPERT_NO_SPLIT': '0'},
    'mmvq_il_narrow_acc': {'STRATA_MMVQ_IL_NARROW_ACC': '0'},
    'hc_q8_16':           {'STRATA_HC_Q8_16': '0'},
    'qsa_scores_16':      {'STRATA_QSA_SCORES_16': '0'},
    'gdn_groups_split':   {'STRATA_GDN_GROUPS_SPLIT': '0'},
    'gdn_ab_wide':        {'STRATA_GDN_AB_WIDE': '0'},
}
ORDER = list(ABLATIONS)


def arm_env(base, arm):
    e = dict(base)
    e.update(ABLATIONS[arm])  # revert the suspect switch(es)
    dump = OUTROOT / arm / 'accept_dump'; dump.mkdir(parents=True, exist_ok=True)
    e['STRATA_DF2_CAND_DUMP'] = str(dump)
    return e


def gen_arm_config(arm):
    scratch = OUTROOT / arm
    (scratch / 'conv-cache').mkdir(parents=True, exist_ok=True)
    (scratch / 'slots').mkdir(parents=True, exist_ok=True)
    out = OUTROOT / f'{arm}.json'
    cmd = [sys.executable, str(CCFG), str(PROD_PROFILE), str(out), '--exe', PROD_EXE,
           '--port', '8290', '--scratch', str(scratch),
           f'--set=--expert-profile-save={scratch}/learned.bin',
           f'--set=--conversation-cache-dir={scratch}/conv-cache']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'candidate_config {arm}: {r.stdout} {r.stderr}')
    cfg = json.loads(out.read_text()); cfg['parallel'] = 1
    out.write_text(json.dumps(cfg, indent=1)); os.chmod(out, 0o600)
    return out, cfg


def deep_body(max_tokens=1024):
    d = json.loads(DEEP_REQ.read_text())
    return {'model': None, 'messages': d['messages'], 'max_tokens': max_tokens,
            'temperature': 0.0, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}


LOOKUP_RE = __import__('re').compile(r'lookup chain: (\d+) of (\d+) windows')


def launch(arm, cfgpath, envfile, env, idx):
    port = 8290; key = env['STRATA_API_KEY']
    unit = f'ab-sw-{arm}-{idx}'; scratch = OUTROOT / arm
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
    cmd = (f"set -a; . {envfile}; set +a; cd {W.PRODDIR} && "
           f"exec /home/soren/.local/bin/strata serve --profile {cfgpath} --output jsonl > {stdout} 2>&1")
    subprocess.run(['systemd-run', '--user', f'--unit={unit}', '--collect', '--no-block',
                    '-p', 'MemoryMax=infinity', '-p', 'MemorySwapMax=infinity', '-p', 'TasksMax=infinity',
                    '--', '/bin/bash', '-c', cmd], check=True, capture_output=True, text=True)
    rec = {'arm': arm, 'idx': idx, 'unit': unit, 'reverts': ABLATIONS[arm]}
    try:
        ok, unkeyed = W.wait_ready(port, key)
        rec['ready'] = ok; rec['unkeyed_status'] = unkeyed
        if not ok:
            rec['error'] = f'not ready: {unkeyed}'
            rec['stdout_tail'] = Path(stdout).read_text(errors='replace')[-1200:] if Path(stdout).exists() else ''
            return rec
        time.sleep(1)
        banner = Path(logpath).read_text(errors='replace') if Path(logpath).exists() else ''
        est = [l for l in banner.splitlines() if 'engine started:' in l]
        if est:
            lav = shlex.split(est[-1].split('engine started:', 1)[1])
            rec['log_exe_matches'] = bool(lav) and os.path.realpath(lav[0]) == os.path.realpath(PROD_EXE)
            dm = [lav[i + 1] for i, x in enumerate(lav) if x == '--draft-model']
            rec['pack_matches_live'] = (dm == [LIVE_PACK])
            rec['log_solo'] = '--batch' not in lav
            if not rec['pack_matches_live'] or not rec['log_exe_matches']:
                rec['error'] = f'pack/exe mismatch dm={dm} exe_ok={rec["log_exe_matches"]}'; rec['ok'] = False
                return rec
        mpid = int(subprocess.run(['systemctl', '--user', 'show', f'{unit}.service', '-p', 'MainPID', '--value'],
                                  capture_output=True, text=True).stdout.strip() or 0)
        epid = W.find_engine(mpid, PROD_EXE)
        if epid:
            eenv = dict(x.split('=', 1) for x in Path(f'/proc/{epid}/environ').read_text(errors='replace').split('\0') if '=' in x)
            # verify the ablated switch value actually landed in the engine env
            rec['revert_verified'] = {k: eenv.get(k) for k in ABLATIONS[arm]}
            rec['revert_ok'] = all(eenv.get(k) == v for k, v in ABLATIONS[arm].items())
            ce = subprocess.run([sys.executable, str(SKILL / 'scripts/check_engine_env.py'), PROD_EXE, str(envfile)],
                                capture_output=True, text=True)
            rec['check_engine_env_rc'] = ce.returncode
        _, models = W.http_json(f'http://127.0.0.1:{port}/v1/models', key=key)
        model = models['data'][0]['id'] if models else None
        rec['model'] = model
        W.chat(port, key, model, 'Warmup: reply with the single word OK.', 16, 0.0)
        # deep-context request (the regression workload)
        off = len(Path(logpath).read_text(errors='replace')) if Path(logpath).exists() else 0
        body = deep_body(1024); body['model'] = model
        st, o = W.http_json(f'http://127.0.0.1:{port}/v1/chat/completions', key=key, body=body, timeout=900)
        deep = {'status': st}
        if o:
            ch = o['choices'][0]; u = o.get('usage', {})
            deep['finish'] = ch.get('finish_reason')
            deep['prompt_tokens'] = u.get('prompt_tokens'); deep['completion_tokens'] = u.get('completion_tokens')
        dec, acc, off = W.parse_new_log(logpath, off)
        deep['decode'] = dec; deep['acc'] = acc
        newtxt = Path(logpath).read_text(errors='replace')[off:] if Path(logpath).exists() else ''
        lm = LOOKUP_RE.search(Path(logpath).read_text(errors='replace'))
        deep['lookup_win'] = [int(lm.group(1)), int(lm.group(2))] if lm else None
        rec['deep'] = deep
        rec['ok'] = bool(dec)
        if not dec:
            rec['error'] = 'no decode line parsed'
    finally:
        subprocess.run(['systemctl', '--user', 'stop', f'{unit}.service'], capture_output=True, text=True)
        subprocess.run(['bash', str(GPUWAIT), str(port), '180'], capture_output=True, text=True)
    return rec


def aggregate(records):
    summ = {}
    for arm in ORDER:
        rs = [r for r in records if r['arm'] == arm and r.get('ok')]
        dec = [r['deep']['decode'][-1][2] for r in rs if r.get('deep', {}).get('decode')]
        accs = []
        for r in rs:
            for a in (r.get('deep', {}).get('acc') or []):
                if a[1]:
                    accs.append(a[0] / a[1])
        summ[arm] = {
            'n': len(dec),
            'decode_tok_s': {'mean': round(statistics.mean(dec), 1) if dec else None,
                             'min': round(min(dec), 1) if dec else None,
                             'max': round(max(dec), 1) if dec else None},
            'accept_ratio': round(statistics.mean(accs), 3) if accs else None,
            'reverts': ABLATIONS[arm],
        }
    ref = summ['control']['decode_tok_s']['mean']
    refa = summ['control']['accept_ratio']
    for arm in ORDER:
        m = summ[arm]['decode_tok_s']['mean']
        summ[arm]['delta_vs_control_pct'] = round((m - ref) / ref * 100, 1) if m and ref else None
        a = summ[arm]['accept_ratio']
        summ[arm]['accept_delta'] = round(a - refa, 3) if a is not None and refa is not None else None
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--launches', type=int, default=3)
    ap.add_argument('--arms', default=None, help='comma subset of arms (default all)')
    a = ap.parse_args()
    global ORDER
    if a.arms:
        ORDER = [x.strip() for x in a.arms.split(',') if x.strip() in ABLATIONS]
        if 'control' not in ORDER:
            ORDER = ['control'] + ORDER
    OUTROOT.mkdir(parents=True, exist_ok=True)
    base, snap = W.load_base_env()
    cfgs = {}; envfiles = {}; envs = {}
    for arm in ORDER:
        cfg, _ = gen_arm_config(arm)
        e = arm_env(base, arm); ef = OUTROOT / f'{arm}.env'; W.write_env_file(e, ef)
        cfgs[arm] = cfg; envfiles[arm] = ef; envs[arm] = e
    if a.plan:
        print('=== PLAN (per-switch solo ablation, deep-context) ===')
        print('prod_exe', PROD_EXE, 'exists', Path(PROD_EXE).exists())
        print('deep_prompt_tokens_approx', len(json.loads(DEEP_REQ.read_text())['messages'][0]['content']) // 3)
        print('arms', ORDER)
        for arm in ORDER:
            print(f'  [{arm}] reverts={ABLATIONS[arm]}')
        # confirm control matches prod snapshot switches for the suspect keys
        print('control_suspect_values', {k: envs['control'].get(k) for k in
              ['STRATA_MOE_BUCKET_NOSPLIT','STRATA_DRAFT_TAIL_SKIP','STRATA_HC_Q8_16','STRATA_MOE_LDS_NT8','STRATA_MOE_GRID_CAP','STRATA_VERIFY_WMMA_DENSE']})
        print('PLAN_OK')
        return
    if a.run:
        records = []
        for i in range(1, a.launches + 1):
            for arm in ORDER:
                print(f'=== LAUNCH {arm} #{i} ===', flush=True)
                try:
                    rec = launch(arm, cfgs[arm], envfiles[arm], envs[arm], i)
                except Exception as e:
                    rec = {'arm': arm, 'idx': i, 'ok': False, 'error': str(e)}
                records.append(rec)
                (OUTROOT / 'records.jsonl').open('a').write(json.dumps(rec) + '\n')
                d = rec.get('deep', {})
                print(json.dumps({'arm': arm, 'idx': i, 'ok': rec.get('ok'), 'revert_ok': rec.get('revert_ok'),
                                  'env_rc': rec.get('check_engine_env_rc'),
                                  'dec': d.get('decode')[-1][2] if d.get('decode') else None,
                                  'acc': d.get('acc'), 'err': rec.get('error')}), flush=True)
                if not rec.get('ok'):
                    print(f'ABORT after failed launch {arm} #{i}: {rec.get("error")}', flush=True)
                    break
            else:
                continue
            break
        summ = aggregate(records)
        (OUTROOT / 'summary.json').write_text(json.dumps(summ, indent=1))
        print('=== SUMMARY ==='); print(json.dumps(summ, indent=1))
        doc = Path('/home/soren/scratch/Strata-merge-p01245/bench/ab-switch'); doc.mkdir(parents=True, exist_ok=True)
        for f in ['records.jsonl', 'summary.json']:
            if (OUTROOT / f).exists():
                shutil.copy2(OUTROOT / f, doc / f)
        shutil.copy2(Path(__file__), doc / 'ab_switch.py')
        print('DOC_COPIED', doc)


if __name__ == '__main__':
    main()
