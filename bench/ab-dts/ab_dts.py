#!/usr/bin/env python3
"""3-arm DRAFT_TAIL_SKIP sweep (6/7/8) with the regression fix PCIE_FETCH_OVERLAP=0 pinned in all arms.

Same prod binary (fork-58aefafc9-p0), deep-context ~222k prompt / 1024 gen / T0, n=3 interleaved.
Isolates DRAFT_TAIL_SKIP on top of the fixed config. Reference arm = dts8 (current prod value).

Usage: ab_dts.py --plan | --run [--launches 3]
"""
import argparse, json, os, shutil, statistics, sys
from pathlib import Path

sys.path.insert(0, '/home/soren/.hermes/cache/scratch')
import ab_window as W
import ab_switch as S

# reuse the proven ab_switch launch/config machinery, retargeted to this sweep
S.ABLATIONS = {
    'dts6': {'STRATA_DRAFT_TAIL_SKIP': '6'},
    'dts7': {'STRATA_DRAFT_TAIL_SKIP': '7'},
    'dts8': {'STRATA_DRAFT_TAIL_SKIP': '8'},
}
S.ORDER = ['dts6', 'dts7', 'dts8']
S.OUTROOT = Path('/home/soren/.hermes/cache/scratch/ab_dts')
OUTROOT = S.OUTROOT
ORDER = S.ORDER
REF = 'dts8'


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
            'DRAFT_TAIL_SKIP': S.ABLATIONS[arm]['STRATA_DRAFT_TAIL_SKIP'],
            'decode_tok_s': {'mean': round(statistics.mean(dec), 1) if dec else None,
                             'min': round(min(dec), 1) if dec else None,
                             'max': round(max(dec), 1) if dec else None},
            'accept_ratio': round(statistics.mean(accs), 3) if accs else None,
        }
    ref = summ[REF]['decode_tok_s']['mean']; refa = summ[REF]['accept_ratio']
    for arm in ORDER:
        m = summ[arm]['decode_tok_s']['mean']
        summ[arm]['delta_vs_dts8_pct'] = round((m - ref) / ref * 100, 1) if m and ref else None
        a = summ[arm]['accept_ratio']
        summ[arm]['accept_delta_vs_dts8'] = round(a - refa, 3) if a is not None and refa is not None else None
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--launches', type=int, default=3)
    a = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    base, snap = W.load_base_env()
    base['STRATA_PCIE_FETCH_OVERLAP'] = '0'  # pin the regression fix in EVERY arm
    cfgs = {}; envfiles = {}; envs = {}
    for arm in ORDER:
        cfg, _ = S.gen_arm_config(arm)
        e = S.arm_env(base, arm); ef = OUTROOT / f'{arm}.env'; W.write_env_file(e, ef)
        cfgs[arm] = cfg; envfiles[arm] = ef; envs[arm] = e
    if a.plan:
        print('=== PLAN (DRAFT_TAIL_SKIP 6/7/8, PCIE_FETCH_OVERLAP=0 pinned) ===')
        print('prod_exe', S.PROD_EXE, 'exists', Path(S.PROD_EXE).exists())
        for arm in ORDER:
            print(f'  [{arm}] DRAFT_TAIL_SKIP={envs[arm]["STRATA_DRAFT_TAIL_SKIP"]} '
                  f'PCIE_FETCH_OVERLAP={envs[arm]["STRATA_PCIE_FETCH_OVERLAP"]}')
        print('PLAN_OK')
        return
    if a.run:
        records = []
        for i in range(1, a.launches + 1):
            for arm in ORDER:
                print(f'=== LAUNCH {arm} #{i} ===', flush=True)
                try:
                    rec = S.launch(arm, cfgs[arm], envfiles[arm], envs[arm], i)
                except Exception as e:
                    rec = {'arm': arm, 'idx': i, 'ok': False, 'error': str(e)}
                rec['pcie_fetch_overlap'] = envs[arm]['STRATA_PCIE_FETCH_OVERLAP']
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
        doc = Path('/home/soren/scratch/Strata-merge-p01245/bench/ab-dts'); doc.mkdir(parents=True, exist_ok=True)
        for f in ['records.jsonl', 'summary.json']:
            if (OUTROOT / f).exists():
                shutil.copy2(OUTROOT / f, doc / f)
        shutil.copy2(Path(__file__), doc / 'ab_dts.py')
        for arm in ORDER:
            for f in [f'{arm}.json', f'{arm}.env']:
                if (OUTROOT / f).exists():
                    shutil.copy2(OUTROOT / f, doc / f)
        print('DOC_COPIED', doc)


if __name__ == '__main__':
    main()
