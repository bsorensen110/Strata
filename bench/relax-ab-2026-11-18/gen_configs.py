#!/usr/bin/env python3
"""Generate per-arm candidate configs for the RDNA4-Ports switch campaign.

Every arm is generated from the live production profile with candidate_config.py, so
args/model/drafter/parallel match production exactly. Only exe/port/log/cwd/slot folder,
the conversation-cache dir and the expert-profile-save path move to arm scratch.
"""
import hashlib, json, os, subprocess, sys

ROOT = '/home/soren/scratch/Strata-merge-0161/bench/relax-ab-2026-11-18'
EXE = '/home/soren/scratch/Strata-merge-0161/build-hip-port/strata'
PROD = '/home/soren/.local/share/strata-upstream-v0.2.10/components/strata-iq3_s.json'
CAND = '/home/soren/.hermes/skills/software-development/strata-hip-install/scripts/candidate_config.py'
PORTS = {'ctl': 8300, 'trt': 8301}
HASHES = f'{ROOT}/configs/generated-hashes.json'


def arm_specs():
    tab = json.load(open(f'{ROOT}/arms.json'))
    out = []
    for i, a in enumerate(tab['arms']):
        # Fast switches first is counter-productive; keep the doc order for a stable narrative.
        out.append((f"ctl_{a['id']}", a, a['baseline']))
        out.append((f"trt_{a['id']}", a, a['arm']))
    return out


def flag_value(args, flag, name):
    positions = [i for i, arg in enumerate(args) if arg == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(args):
        raise SystemExit(f'preflight: {name} must have exactly one {flag} value')
    return args[positions[0] + 1]


def assert_cache_args(args, arm, name):
    budget = str(arm['expert_cache_budget_slots'])
    if flag_value(args, '--expert-cache', name) != budget:
        raise SystemExit(f'preflight: {name} must pin --expert-cache budget {budget}')
    if '--expert-cache-quota' in args:
        raise SystemExit(f'preflight: {name} must not contain --expert-cache-quota')
    if args.count('--expert-cache-per-layer') != 1:
        raise SystemExit(f'preflight: {name} must retain exactly one --expert-cache-per-layer flag')


def assert_preflight():
    specs = arm_specs()
    report_paths = set()
    try:
        hashes = json.load(open(HASHES))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f'preflight: generated config hashes unavailable: {e}')
    for name, arm, _ in specs:
        path = f'{ROOT}/configs/{name}.json'
        try:
            raw = open(path, 'rb').read()
            cfg = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f'preflight: cannot read {path}: {e}')
        expected_hash = hashes.get(name)
        actual_hash = hashlib.sha256(raw).hexdigest()
        if expected_hash != actual_hash:
            raise SystemExit(f'preflight: stale config hash for {name}; rerun gen_configs.py')
        report_path = cfg.get('env', {}).get('STRATA_DISPATCH_REPORT')
        expected_report_path = f'{ROOT}/results/{name}.dispatch.json'
        if report_path != expected_report_path or report_path in report_paths:
            raise SystemExit(f'preflight: {name} has missing, non-unique, or unexpected dispatch-report path')
        report_paths.add(report_path)
        args = cfg.get('args') or []
        assert_cache_args(args, arm, name)
        if cfg.get('env', {}).get('STRATA_WINDOW_WARM') != '0':
            raise SystemExit(f'preflight: {name} must set STRATA_WINDOW_WARM=0')
        user_args = cfg.get('user_args') or []
        for flag, value in (('--lookup-max-t', '16'), ('--lookup-sizes', '10,12,14,16'),
                            ('--lookup-chain', '6')):
            positions = [i for i, arg in enumerate(user_args) if arg == flag]
            if len(positions) != 1 or positions[0] + 1 >= len(user_args) or \
                    user_args[positions[0] + 1] != value:
                raise SystemExit(f'preflight: {name} must retain {flag} {value}')
        if cfg.get('parallel') != 1:
            raise SystemExit(f'preflight: {name} must use parallel=1')
        if cfg.get('drafting') != 'dflash2':
            raise SystemExit(f'preflight: {name} must pin drafting to dflash2')
    return sorted(report_paths)


def main():
    only = sys.argv[1:] or None
    specs = arm_specs()
    ids = {a['id'] for a in json.load(open(f'{ROOT}/arms.json'))['arms']}
    for a in (only or []):
        if a not in ids:
            sys.exit(f'unknown arm id {a}')
    made = []
    for name, a, value in specs:
        if only and a['id'] not in only:
            continue
        role = 'ctl' if name.startswith('ctl_') else 'trt'
        scratch = f'{ROOT}/configs/{name}'
        os.makedirs(scratch, exist_ok=True)
        conv = f'{scratch}/conversation-cache'
        os.makedirs(conv, exist_ok=True)
        out = f'{ROOT}/configs/{name}.json'
        exe = (a.get('exe_ctl') if role == 'ctl' else a.get('exe_arm')) or EXE
        cmd = [sys.executable, CAND, PROD, out, '--exe', exe, '--port', str(PORTS[role]), '--scratch', scratch,
               f'--set=--conversation-cache-dir={conv}',
               f'--set=--expert-profile-save={scratch}/profile-learned.bin']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'candidate_config failed for {name}: {r.stdout}{r.stderr}')
        cfg = json.load(open(out))
        # Reserve identical VRAM headroom for the relaxed W6 projection packs in every arm.
        quota_positions = [i for i, arg in enumerate(cfg['args'])
                           if arg == '--expert-cache-quota']
        if (len(quota_positions) != 1 or quota_positions[0] + 1 >= len(cfg['args']) or
                cfg['args'][quota_positions[0] + 1] != 'auto'):
            sys.exit(f'{name}: expected exactly one well-formed --expert-cache-quota auto pair')
        del cfg['args'][quota_positions[0]:quota_positions[0] + 2]
        cache_positions = [i for i, arg in enumerate(cfg['args']) if arg == '--expert-cache']
        if len(cache_positions) != 1 or cache_positions[0] + 1 >= len(cfg['args']):
            sys.exit(f'{name}: production profile has no unique valid --expert-cache value')
        cfg['args'][cache_positions[0] + 1] = str(a['expert_cache_budget_slots'])
        # serve.server rebuilds the drafter argv from this key; 'auto' resolves to "drafting off",
        # while the production wrapper resolves it to DFlash2. Pin the production-resolved mode.
        cfg['drafting'] = 'dflash2'
        cfg['env'] = {
            'STRATA_MANIFEST': f'{ROOT}/results/{name}.manifest.json',
            'STRATA_DISPATCH_REPORT': f'{ROOT}/results/{name}.dispatch.json',
            'STRATA_WINDOW_WARM': '0',
        }
        for k, v in (a.get('pin') or {}).items():          # pinned on BOTH sides: the base of the comparison
            cfg['env'][k] = str(v)
        for k, v in ((a.get('pin_ctl') if role == 'ctl' else a.get('pin_arm')) or {}).items():
            cfg['env'][k] = str(v)
        if a.get('switch'):
            cfg['env'][a['switch']] = str(value)   # binary arms pin no switch at all
        extra = a.get('args_ctl' if role == 'ctl' else 'args_arm') or []
        if extra:
            cfg['args'] = list(cfg['args']) + [str(x) for x in extra]
        user_args = cfg.get('user_args')
        required_user_args = (
            ('--lookup-max-t', '16'),
            ('--lookup-sizes', '10,12,14,16'),
            ('--lookup-chain', '6'),
            ('--spec', '8'),
            ('--draft-conf', '0.35'),
        )
        if not isinstance(user_args, list):
            sys.exit(f'{name}: production profile has no user_args list')
        for flag, value in required_user_args:
            positions = [pos for pos, arg in enumerate(user_args) if arg == flag]
            if len(positions) != 1:
                sys.exit(f'{name}: production user_args missing {flag} {value}')
            pos = positions[0]
            if pos + 1 >= len(user_args) or user_args[pos + 1] != value:
                sys.exit(f'{name}: production user_args must retain {flag} {value}')
        with open(out, 'w') as fh:
            json.dump(cfg, fh, indent=1)
        os.chmod(out, 0o600)
        made.append(name)
    hashes = {}
    if only and os.path.exists(HASHES):
        with open(HASHES) as fh:
            hashes.update(json.load(fh))
    for name in made:
        with open(f'{ROOT}/configs/{name}.json', 'rb') as fh:
            hashes[name] = hashlib.sha256(fh.read()).hexdigest()
    with open(HASHES, 'w') as fh:
        json.dump(hashes, fh, indent=1, sort_keys=True)
    os.chmod(HASHES, 0o600)
    paths = assert_preflight()
    print(f'generated {len(made)} configs: ' + ', '.join(made))
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
