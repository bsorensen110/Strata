#!/usr/bin/env python3
"""Generate per-arm candidate configs for the RDNA4-Ports switch campaign.

Every arm is generated from the live production profile with candidate_config.py, so
args/model/drafter/parallel match production exactly. Only exe/port/log/cwd/slot folder,
the conversation-cache dir and the expert-profile-save path move to arm scratch.
"""
import json, os, subprocess, sys

ROOT = '/home/soren/Strata-latest-20261112-182859/bench/ports-ab-2026-11-12'
EXE = '/home/soren/Strata-latest-20261112-182859/build-hip/strata'
PROD = '/home/soren/.local/share/strata-upstream-v0.2.10/components/strata-iq3_s.json'
CAND = '/home/soren/.hermes/skills/software-development/strata-hip-install/scripts/candidate_config.py'
PORTS = {'ctl': 8300, 'trt': 8301}


def arm_specs():
    tab = json.load(open(f'{ROOT}/arms.json'))
    out = []
    for i, a in enumerate(tab['arms']):
        # Fast switches first is counter-productive; keep the doc order for a stable narrative.
        out.append((f"ctl_{a['id']}", a, a['baseline']))
        out.append((f"trt_{a['id']}", a, a['arm']))
    return out


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
        cmd = [sys.executable, CAND, PROD, out, '--exe', EXE, '--port', str(PORTS[role]), '--scratch', scratch,
               f'--set=--conversation-cache-dir={conv}',
               f'--set=--expert-profile-save={scratch}/profile-learned.bin']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'candidate_config failed for {name}: {r.stdout}{r.stderr}')
        cfg = json.load(open(out))
        # serve.server rebuilds the drafter argv from this key; 'auto' resolves to "drafting off",
        # while the production wrapper resolves it to DFlash2. Pin the production-resolved mode.
        cfg['drafting'] = 'dflash2'
        cfg['env'] = {
            'STRATA_MANIFEST': f'{ROOT}/results/{name}.manifest.json',
            a['switch']: str(value),
        }
        cfg['user_args'] = ['--spec', '8', '--draft-conf', '0.35'] if cfg.get('user_args') else []
        with open(out, 'w') as fh:
            json.dump(cfg, fh, indent=1)
        os.chmod(out, 0o600)
        made.append(name)
    print(f'generated {len(made)} configs: ' + ', '.join(made))


if __name__ == '__main__':
    main()
