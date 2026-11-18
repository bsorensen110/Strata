#!/usr/bin/env python3
"""Generate per-arm candidate configs for Campaign 7 (ports-ab-2026-11-17-win16).

Supports both:
- Same-binary env-switch arms (default ctl_exe = trt_exe = fork-68872b2d1/strata), and
- Binary + env-switch arms with per-arm ctl_exe / trt_exe and ctl_env / trt_env dicts,
  while remaining compatible with legacy switch/baseline/arm/pin/pin_ctl/pin_arm fields.
"""
import json, os, subprocess, sys

ROOT = '/home/soren/Strata-latest-20261112-182859/bench/ports-ab-2026-11-17-win16'
DEFAULT_EXE = '/home/soren/.local/share/strata-upstream-v0.2.10/components/engine/fork-68872b2d1/strata'
EXE = DEFAULT_EXE
PROD = '/home/soren/.local/share/strata-upstream-v0.2.10/components/strata-iq3_s.json'
CAND = '/home/soren/.hermes/skills/software-development/strata-hip-install/scripts/candidate_config.py'
BASE_PORT = 8300


def resolve_exe(arm, role, default_exe=DEFAULT_EXE):
    if role == 'ctl':
        return arm.get('ctl_exe') or arm.get('exe_ctl') or arm.get('exe') or default_exe
    return arm.get('trt_exe') or arm.get('exe_arm') or arm.get('exe') or default_exe


def resolve_port(arm, idx, role):
    key = 'ctl_port' if role == 'ctl' else 'trt_port'
    if arm.get(key) is not None:
        return int(arm[key])
    return BASE_PORT + 2 * idx + (0 if role == 'ctl' else 1)


def resolve_env(arm, role, name):
    env = {'STRATA_MANIFEST': f'{ROOT}/results/{name}.manifest.json'}
    for k, v in (arm.get('pin') or {}).items():
        env[str(k)] = str(v)
    for k, v in ((arm.get('pin_ctl') if role == 'ctl' else arm.get('pin_arm')) or {}).items():
        env[str(k)] = str(v)
    for k, v in ((arm.get('ctl_env') if role == 'ctl' else arm.get('trt_env')) or {}).items():
        env[str(k)] = str(v)
    if arm.get('switch'):
        val = arm.get('baseline') if role == 'ctl' else arm.get('arm')
        if val is not None:
            env[str(arm['switch'])] = str(val)
    if env.get('STRATA_DF2_HEAD') == 'pruned':
        sys.exit(f'refusing {name}: STRATA_DF2_HEAD=pruned is forbidden')
    return env


def arm_specs():
    tab = json.load(open(f'{ROOT}/arms.json'))
    default_exe = tab.get('default_exe') or DEFAULT_EXE
    out = []
    seen_ports = set()
    for idx, a in enumerate(tab['arms']):
        for role in ('ctl', 'trt'):
            name = f"{role}_{a['id']}"
            port = resolve_port(a, idx, role)
            if port in seen_ports:
                sys.exit(f'duplicate port {port} on {name}')
            seen_ports.add(port)
            exe = resolve_exe(a, role, default_exe)
            out.append((name, a, role, port, exe))
    return out


def main():
    only = sys.argv[1:] or None
    specs = arm_specs()
    ids = {a['id'] for a in json.load(open(f'{ROOT}/arms.json'))['arms']}
    for a in (only or []):
        if a not in ids:
            sys.exit(f'unknown arm id {a}')
    made = []
    for name, a, role, port, exe in specs:
        if only and a['id'] not in only:
            continue
        if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
            sys.exit(f'missing or non-executable engine binary for {name}: {exe}')
        scratch = f'{ROOT}/configs/{name}'
        os.makedirs(scratch, exist_ok=True)
        conv = f'{scratch}/conversation-cache'
        os.makedirs(conv, exist_ok=True)
        out = f'{ROOT}/configs/{name}.json'
        cmd = [sys.executable, CAND, PROD, out, '--exe', exe, '--port', str(port), '--scratch', scratch,
               f'--set=--conversation-cache-dir={conv}',
               f'--set=--expert-profile-save={scratch}/profile-learned.bin']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'candidate_config failed for {name}: {r.stdout}{r.stderr}')
        cfg = json.load(open(out))
        # serve.server rebuilds the drafter argv from this key; 'auto' resolves to "drafting off",
        # while the production wrapper resolves it to DFlash2. Pin the production-resolved mode.
        cfg['drafting'] = 'dflash2'
        cfg['env'] = resolve_env(a, role, name)
        extra = a.get('args_ctl' if role == 'ctl' else 'args_arm') or []
        if extra:
            cfg['args'] = list(cfg['args']) + [str(x) for x in extra]
        cfg['user_args'] = ['--spec', '8', '--draft-conf', '0.35'] if cfg.get('user_args') else []
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as fh:
            json.dump(cfg, fh, indent=1)
        os.chmod(out, 0o600)
        made.append(name)
    print(f'generated {len(made)} configs: ' + ', '.join(made))


if __name__ == '__main__':
    main()
