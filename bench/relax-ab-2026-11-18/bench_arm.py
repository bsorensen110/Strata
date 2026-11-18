#!/usr/bin/env python3
"""Measure one campaign arm; write results/<arm>.json (+ .texts.json for parity diffs).

Workload is fixed across arms and mirrors production: same model, pack, drafter, parallel=1,
spec/draft-conf, expert cache, conversation cache policy. Trials:
  smoke    (256 tok, temp 0)   - readiness / sanity, not a timing sample
  p4k      (~3.9k tok prompt)  - cold prompt processing, includes its decode
  greedy1  (512 tok, temp 0)   - cold solo decode (first time this prompt is seen)
  greedy2  (512 tok, temp 0)   - warm solo decode
  burst    (2 x 256, temp 0)   - concurrent decode, aggregate from client usage tokens over shared wall
  wide     (512 tok, temp 0)   - repeated 9-16 token suffix/lookup window workload
Greedy markers are identical across arms so the parity hashes are comparable; the smoke
marker carries the arm name and is excluded from parity.
"""
import concurrent.futures, hashlib, json, os, random, re, subprocess, sys, time, urllib.request
from route_evidence import (classify_route, read_dispatch_report,
                            iu8_route_status, parse_iu8_lines,
                            validate_expert_cache_slots, validate_iu8_gate)

ROOT = '/home/soren/scratch/Strata-merge-0161/bench/relax-ab-2026-11-18'
NAME = sys.argv[1]
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ''
cfg = json.load(open(f'{ROOT}/configs/{NAME}.json'))
port = cfg['port']
url = f'http://127.0.0.1:{port}'
log = cfg['log']
serve_out = f'{ROOT}/configs/{NAME}/serve.out'
unit = f'portsab-{NAME}'
arm_id = NAME.removeprefix('ctl_').removeprefix('trt_')
arms = json.load(open(f'{ROOT}/arms.json'))['arms']
expected_cache_slots = next(a['expert_cache_slots'] for a in arms if a['id'] == arm_id)
requested_cache_budget = next(a['expert_cache_budget_slots'] for a in arms if a['id'] == arm_id)
cache_args = cfg.get('args') or []
cache_positions = [i for i, arg in enumerate(cache_args) if arg == '--expert-cache']
if len(cache_positions) != 1 or cache_positions[0] + 1 >= len(cache_args):
    sys.exit(f'{NAME}: expected exactly one --expert-cache value')
configured_cache_budget = int(cache_args[cache_positions[0] + 1])
if configured_cache_budget != requested_cache_budget:
    sys.exit(f'{NAME}: config cache budget mismatch: expected {requested_cache_budget}, '
             f'requested {configured_cache_budget}')

# API key: read in-process from the production env file, never printed.
key = None
for line in open('/home/soren/.config/strata-upstream-v0.2.10/production.env'):
    line = line.strip()
    if line.startswith('STRATA_API_KEY') and '=' in line:
        key = line.split('=', 1)[1].strip().strip('"').strip("'")
if not key:
    sys.exit('no api key found in production.env')

start = time.monotonic()
model = None
while time.monotonic() - start < 600:
    try:
        q = urllib.request.Request(url + '/v1/models', headers={'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(q, timeout=5) as r:
            model = json.loads(r.read())['data'][0]['id']
        break
    except Exception:
        pass
    # fail fast instead of waiting out the timeout when the engine died at startup
    alive = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip()
    try:
        startup = open(log, errors='replace').read()[-40000:]
    except OSError:
        startup = ''
    if alive != 'active' or 'out of memory' in startup:
        tail = [l for l in startup.splitlines() if 'out of memory' in l or 'error' in l.lower()][-3:]
        sys.exit(f'{NAME}: engine not running during readiness (state={alive}) :: {tail}')
    time.sleep(2)
if model is None:
    sys.exit(f'{NAME}: readiness timeout')

# Effective process state, not the profile JSON.
main = subprocess.check_output(['systemctl', '--user', 'show', '-p', 'MainPID', '--value', unit], text=True).strip()
if not main or main == '0':
    sys.exit(f'{NAME}: wrapper MainPID missing')
ep = None
for p in subprocess.check_output(['pgrep', '-P', main], text=True).split():
    try:
        if os.path.basename(os.readlink(f'/proc/{p}/exe')) == 'strata':
            ep = int(p)
            break
    except OSError:
        pass
if ep is None:
    sys.exit(f'{NAME}: engine child not found')
argv = [x.decode() for x in open(f'/proc/{ep}/cmdline', 'rb').read().split(b'\0') if x]
procenv = dict(l.decode(errors='replace').split('=', 1) for l in open(f'/proc/{ep}/environ', 'rb').read().split(b'\0') if b'=' in l)
for k, v in cfg.get('env', {}).items():
    if procenv.get(k) != str(v):
        sys.exit(f'{NAME}: env mismatch for {k}: engine has {procenv.get(k)!r}, arm wants {v!r}')
for flag in ('--kv-resident', '--kv-host-grow'):
    if flag not in argv:
        sys.exit(f'{NAME}: missing fixed production argument {flag}')
if '--batch' in argv and argv[argv.index('--batch') + 1] != str(cfg.get('parallel', 1)):
    sys.exit(f'{NAME}: batch width mismatch')

initial_log = ''
if os.path.exists(serve_out):
    initial_log += open(serve_out, errors='replace').read()
launch_log = open(log, errors='replace').read() if os.path.exists(log) else ''
initial_log += launch_log
if 'dflash2' not in initial_log.lower():
    sys.exit(f'{NAME}: DFlash2 banner missing from startup log')

cache_status, actual_cache_slots = validate_expert_cache_slots(
    initial_log.splitlines(), expected_cache_slots)
if cache_status != 'ok':
    kind = 'missing' if cache_status == 'cache_evidence_missing' else 'mismatch'
    actual_text = 'none' if actual_cache_slots is None else str(actual_cache_slots)
    print(f'CACHE_GATE_FAIL kind={kind} expected_actual={expected_cache_slots} '
          f'actual={actual_text} budget={requested_cache_budget}', flush=True)
    sys.exit(2)

def post(prompt, max_tokens=256, temp=0.0):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': temp,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.monotonic() - t
    u = d.get('usage') or {}
    ch = d['choices'][0]
    txt = ch['message'].get('content') or ''
    ct, pt = u.get('completion_tokens'), u.get('prompt_tokens')
    if type(ct) is not int or ct <= 0 or type(pt) is not int or pt <= 0 or not txt:
        raise RuntimeError('missing usage or empty completion')
    return {'prompt_tokens': pt, 'completion_tokens': ct, 'finish': ch.get('finish_reason'),
            'wall_s': round(dt, 3), 'wall_tok_s': round(ct / dt, 2), 'text': txt,
            'hash': hashlib.sha256(txt.encode()).hexdigest()}

smoke = post(f'[portsab {NAME} smoke] Reply with only the result of 19+23.', 256, 0.0)
if smoke['finish'] not in ('stop', 'length'):
    sys.exit(f'{NAME}: smoke failed {smoke["finish"]}')
if smoke['completion_tokens'] > 8:
    sys.exit(f'{NAME}: smoke exceeded 8 tokens ({smoke["completion_tokens"]}); wide request not run')


def wide_prompt():
    pair_id = arm_id
    round_no = int(SUFFIX.removeprefix('_r') or '1')
    seed = int.from_bytes(hashlib.sha256(f'campaign9:{pair_id}:{round_no}'.encode()).digest()[:8], 'big')
    rng = random.Random(seed)
    colors = ('amber', 'copper', 'indigo', 'silver', 'verdant', 'ochre', 'violet', 'saffron')
    animals = ('heron', 'badger', 'fox', 'otter', 'raven', 'lynx', 'marten', 'crane')
    shapes = ('spiral', 'hexagon', 'wave', 'diamond', 'crescent', 'lattice', 'ribbon', 'whorl')
    clauses = []
    for _ in range(8):
        clauses.append(f"the {rng.choice(colors)} {rng.choice(animals)} traces a "
                       f"{rng.choice(shapes)} path")
    block = f'At first light of round {round_no}, ' + ', while '.join(clauses) + '.'
    return ('Continue the pattern below by repeating the exact block over and over until the length limit. '
            'Do not explain or add anything else. The block is:\n' + (block + '\n') * 8)


def engine_log_from(offset=0):
    try:
        with open(log, 'rb') as fh:
            fh.seek(offset)
            return fh.read().decode(errors='replace')
    except OSError:
        return ''


def grab(pattern, text, limit=8):
    return [m.group(0) for m in re.finditer(pattern, text)][-limit:]


wide_prompt_text = wide_prompt()
wide_log_offset = os.path.getsize(log) if os.path.exists(log) else 0
wide = post(wide_prompt_text, 512, 0.0)
wide_log = engine_log_from(wide_log_offset)
wide_request_lines = [line for line in wide_log.splitlines()
                      if 'suffix drafts:' in line.lower() or 'lookup chain:' in line.lower()]
wide_generated_lines = grab(r'generated in \d+ ms[^\n]*', wide_log)

PREFILL_PROMPT = ('Explain how a radix tree stores strings, including insertion and prefix lookup. ' * 280)
prefill = post('[ports-ab p4k]\n' + PREFILL_PROMPT + '\nFinish with: done.', 256, 0.0)

GREEDY_PROMPT = ('[ports-ab greedy] Write Python code for a deterministic LRU cache with put/get and tests. Code only.')
greedy = []
for i in range(2):
    g = post(GREEDY_PROMPT, 512, 0.0)
    if g['completion_tokens'] < 512:
        sys.exit(f'{NAME}: greedy trial {i+1} truncated at {g["completion_tokens"]} tokens')
    greedy.append(g)

BURST_PROMPTS = [
    '[ports-ab burst-a] Write a Python function for Dijkstra shortest paths with heapq and tests. Code only.',
    '[ports-ab burst-b] Write a Python function for topological sort with cycle detection and tests. Code only.',
]
t0 = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    burst = list(ex.map(lambda p: post(p, 256, 0.0), BURST_PROMPTS))
burst_s = time.monotonic() - t0
burst_tok = sum(r['completion_tokens'] for r in burst)
if any(r['completion_tokens'] != 256 for r in burst):
    sys.exit(f'{NAME}: burst did not return 2 x 256 tokens')

# SIGTERM is sent to the server wrapper by systemd; its handler closes the engine with STOP/QUIT.
# The direct engine signal handler exits by re-raising the signal and does not run atexit.
stop_error = None
try:
    stopped = subprocess.run(['systemctl', '--user', 'stop', f'{unit}.service'],
                             capture_output=True, text=True, timeout=60)
    if stopped.returncode != 0:
        stop_error = f'service stop failed ({stopped.returncode}): {stopped.stderr.strip()}'
except (OSError, subprocess.TimeoutExpired) as e:
    stop_error = f'service stop failed: {type(e).__name__}: {e}'

report_path = cfg.get('env', {}).get('STRATA_DISPATCH_REPORT', '')
report_error = None
dispatch = None
for attempt in range(4):
    try:
        dispatch = read_dispatch_report(report_path)
        if dispatch is not None:
            break
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
        if isinstance(e, ValueError) and str(e) == 'invalid dispatch report shape':
            report_error = f'dispatch-report invalid shape: {e}'
        elif isinstance(e, json.JSONDecodeError):
            report_error = f'dispatch-report invalid shape: invalid JSON: {e}'
        else:
            report_error = f'dispatch-report unreadable: {type(e).__name__}: {e}'
        break
    if attempt < 3:
        time.sleep(1)
if dispatch is None and report_error is None:
    report_error = 'dispatch-report missing: evidence channel failure'

launch_log = open(log, errors='replace').read() if os.path.exists(log) else ''


slots_line = grab(r'expert cache \d+ slots[^\n]*', initial_log, 1)
relaxed_projection = re.findall(r'HIP projection [^\n]*relaxed W6 ready before capture[^\n]*', launch_log)
route_expected = cfg.get('env', {}).get('STRATA_HIP_MOE_X4') == '1'
x4_captures = dispatch['x4_captures'] if dispatch else {'n': 0, 'entries': []}
wide_launches = dispatch['wide_launches'] if dispatch else {'k0': 0, 'k1': 0, 'k2': 0}
wide_launches_by_t = dispatch['wide_launches_by_t'] if dispatch else {}
route_valid, invalid_reason = classify_route(route_expected, dispatch, wide_request_lines)
route_engaged = route_expected and route_valid
if report_error:
    route_valid, invalid_reason = False, report_error
elif route_valid and (wide['finish'] != 'length' or wide['completion_tokens'] < 512):
    route_valid, invalid_reason = False, 'wide workload terminated early (pattern abandoned)'
iu8_evidence = parse_iu8_lines(launch_log.splitlines())
iu8_evidence['iu8_route_status'] = iu8_route_status(iu8_evidence, cfg.get('env', {}))
iu8_valid, iu8_invalid_reason = validate_iu8_gate(cfg.get('env', {}), iu8_evidence)
if not iu8_valid:
    route_valid, invalid_reason = False, iu8_invalid_reason

manifest_path = cfg.get('env', {}).get('STRATA_MANIFEST')
if not manifest_path:
    raise RuntimeError(f'{NAME}: STRATA_MANIFEST is missing from launch config')
try:
    with open(manifest_path) as fh:
        manifest = json.load(fh)
except (OSError, json.JSONDecodeError) as e:
    raise RuntimeError(f'{NAME}: cannot read launch manifest {manifest_path}: {e}') from e
manifest.update(iu8_evidence)
try:
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=1)
except OSError as e:
    raise RuntimeError(f'{NAME}: cannot write launch manifest {manifest_path}: {e}') from e
rec = {
    'arm': NAME,
    'round': SUFFIX or 'r1',
    'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'exe': cfg['exe'],
    'binary_sha256': hashlib.sha256(open(cfg['exe'], 'rb').read()).hexdigest(),
    'commit': subprocess.check_output(['git', '-C', '/home/soren/scratch/Strata-merge-0161', 'rev-parse', '--short', 'HEAD'], text=True).strip(),
    'engine_pid': ep,
    'arm_env': cfg.get('env', {}),
    'user_args': cfg.get('user_args', []),
    'parallel': cfg.get('parallel'),
    'route_expected': route_expected,
    'route_engaged': route_engaged,
    'timing_valid': route_valid,
    'expert_cache_budget_slots': requested_cache_budget,
    'expected_expert_cache_slots': expected_cache_slots,
    'actual_expert_cache_slots': actual_cache_slots,
    **iu8_evidence,
    'x4_captures': x4_captures,
    'wide_launches': wide_launches,
    'wide_launches_by_t': wide_launches_by_t,
    'wide_request_lines': wide_request_lines,
    'route_evidence': {
        'route_expected': route_expected,
        'route_engaged': route_engaged,
        'valid': route_valid,
        'invalid_reason': invalid_reason,
        'evidence_channel_failure': bool(report_error),
        'window_warm': cfg.get('env', {}).get('STRATA_WINDOW_WARM'),
        'wide_request_order': ['smoke', 'wide', 'prefill', 'greedy1', 'greedy2', 'burst'],
        'expected_expert_cache_slots': expected_cache_slots,
        'actual_expert_cache_slots': actual_cache_slots,
        **iu8_evidence,
        'x4_captures': x4_captures,
        'x4_captures_by_scope': dispatch.get('x4_captures_by_scope', []) if dispatch else [],
        'x4_wide_scope_note_count': dispatch.get('x4_wide_scope_note_count', 0) if dispatch else 0,
        'wide_launches': wide_launches,
        'wide_launches_by_t': wide_launches_by_t,
        'wide_request_lines': wide_request_lines,
        'wide_request_line_count': len(wide_request_lines),
        'wide_window_log_lines': wide_request_lines,
        'relaxed_projection_pack_count': len(relaxed_projection),
        'relaxed_projection_lines': relaxed_projection,
        'dispatch_report_path': report_path,
        'dispatch_report_launches': dispatch.get('launches', {}) if dispatch else {},
        'dispatch_report_counts_preview': dispatch.get('counts_preview', []) if dispatch else [],
        'graceful_stop_error': stop_error,
    },
    'smoke': {k: smoke[k] for k in ('prompt_tokens', 'completion_tokens', 'finish', 'wall_s', 'wall_tok_s')},
    'prefill': {k: prefill[k] for k in ('prompt_tokens', 'completion_tokens', 'finish', 'wall_s', 'wall_tok_s')},
    'greedy': [{k: g[k] for k in ('completion_tokens', 'finish', 'wall_s', 'wall_tok_s', 'hash')} for g in greedy],
    'burst': {'wall_s': round(burst_s, 3), 'completion_tokens': burst_tok,
              'aggregate_tok_s': round(burst_tok / burst_s, 2),
              'slots': [{k: r[k] for k in ('completion_tokens', 'wall_s', 'wall_tok_s', 'hash')} for r in burst]},
    'wide': {k: wide[k] for k in ('completion_tokens', 'finish', 'wall_s', 'wall_tok_s', 'hash')},
    'parity': {'greedy_cold': greedy[0]['hash'], 'greedy_warm': greedy[1]['hash'],
               'burst_a': burst[0]['hash'], 'burst_b': burst[1]['hash']},
    'engine_decode_512': [m.group(1) for m in re.finditer(r'512 generated in \d+ ms \(([0-9.]+) tok/s\)', launch_log)][:2],
    'engine_log_lines': {
        'generated': grab(r'generated in \d+ ms[^\n]*', launch_log),
        'prompt_read': grab(r'prompt \d+ tokens = [^\n]*', launch_log),
        'drafts': grab(r'drafts accepted \d+ of \d+', launch_log),
        'wide_generated': wide_generated_lines,
    },
    'expert_cache': slots_line,
    'expert_cache_slots': actual_cache_slots,
    'actual_expert_cache_slots': actual_cache_slots,
}
if not route_valid:
    rec['invalid_reason'] = invalid_reason
if stop_error:
    rec['route_evidence']['graceful_stop_error'] = stop_error
os.makedirs(f'{ROOT}/results', exist_ok=True)
with open(f'{ROOT}/results/{NAME}{SUFFIX}.json', 'w') as fh:
    json.dump(rec, fh, indent=1)
with open(f'{ROOT}/results/{NAME}{SUFFIX}.texts.json', 'w') as fh:
    json.dump({'greedy': [g['text'] for g in greedy], 'burst': [r['text'] for r in burst],
               'prefill': prefill['text']}, fh, indent=1)
print(json.dumps({'arm': NAME, 'greedy_tok_s': [g['wall_tok_s'] for g in greedy],
                  'burst_agg': rec['burst']['aggregate_tok_s'], 'prefill_tok_s': prefill['wall_tok_s'],
                  'prefill_prompt_tokens': prefill['prompt_tokens'],
                  'wide_tok_s': wide['wall_tok_s'], 'hashes': rec['parity'], 'cache': rec['expert_cache'],
                  'expected_cache_slots': expected_cache_slots, 'actual_cache_slots': actual_cache_slots,
                  'route_valid': route_valid, 'invalid_reason': invalid_reason}))
