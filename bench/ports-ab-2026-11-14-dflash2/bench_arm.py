#!/usr/bin/env python3
"""Measure one campaign arm; write results/<arm>.json (+ .texts.json for parity diffs).

Workload is fixed across arms and mirrors production: same model, pack, drafter, parallel=2,
spec/draft-conf, expert cache, conversation cache policy. Trials:
  smoke    (256 tok, temp 0)   - readiness / sanity, not a timing sample
  p4k      (~3.9k tok prompt)  - cold prompt processing, includes its decode
  greedy1  (512 tok, temp 0)   - cold solo decode (first time this prompt is seen)
  greedy2  (512 tok, temp 0)   - warm solo decode
  burst    (2 x 256, temp 0)   - concurrent decode, aggregate from client usage tokens over shared wall
Greedy markers are identical across arms so the parity hashes are comparable; the smoke
marker carries the arm name and is excluded from parity.
"""
import concurrent.futures, hashlib, json, os, re, subprocess, sys, time, urllib.request

ROOT = '/home/soren/Strata-latest-20261112-182859/bench/ports-ab-2026-11-14-dflash2'
NAME = sys.argv[1]
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ''
cfg = json.load(open(f'{ROOT}/configs/{NAME}.json'))
port = cfg['port']
url = f'http://127.0.0.1:{port}'
log = cfg['log']
serve_out = f'{ROOT}/configs/{NAME}/serve.out'
unit = f'portsab-{NAME}'

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
for p in (serve_out, log):
    if os.path.exists(p):
        initial_log += open(p, errors='replace').read()
if 'dflash2' not in initial_log.lower():
    sys.exit(f'{NAME}: DFlash2 banner missing from startup log')
off = os.path.getsize(log) if os.path.exists(log) else 0

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

newlog = open(log, errors='replace').read()[off:] if os.path.exists(log) else ''


def grab(pattern, text, limit=8):
    return [m.group(0) for m in re.finditer(pattern, text)][-limit:]


slots_line = grab(r'expert cache \d+ slots[^\n]*', initial_log, 1) or grab(r'\d+ slots \([^\n]*', initial_log, 1)
rec = {
    'arm': NAME,
    'round': SUFFIX or 'r1',
    'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'exe': cfg['exe'],
    'binary_sha256': hashlib.sha256(open(cfg['exe'], 'rb').read()).hexdigest(),
    'commit': subprocess.check_output(['git', '-C', '/home/soren/Strata-latest-20261112-182859', 'rev-parse', '--short', 'HEAD'], text=True).strip(),
    'engine_pid': ep,
    'arm_env': cfg.get('env', {}),
    'parallel': cfg.get('parallel'),
    'smoke': {k: smoke[k] for k in ('prompt_tokens', 'completion_tokens', 'finish', 'wall_s', 'wall_tok_s')},
    'prefill': {k: prefill[k] for k in ('prompt_tokens', 'completion_tokens', 'finish', 'wall_s', 'wall_tok_s')},
    'greedy': [{k: g[k] for k in ('completion_tokens', 'finish', 'wall_s', 'wall_tok_s', 'hash')} for g in greedy],
    'burst': {'wall_s': round(burst_s, 3), 'completion_tokens': burst_tok,
              'aggregate_tok_s': round(burst_tok / burst_s, 2),
              'slots': [{k: r[k] for k in ('completion_tokens', 'wall_s', 'wall_tok_s', 'hash')} for r in burst]},
    'parity': {'greedy_cold': greedy[0]['hash'], 'greedy_warm': greedy[1]['hash'],
               'burst_a': burst[0]['hash'], 'burst_b': burst[1]['hash']},
    'engine_decode_512': [m.group(1) for m in re.finditer(r'512 generated in \d+ ms \(([0-9.]+) tok/s\)', newlog)][:2],
    'engine_log_lines': {
        'generated': grab(r'generated in \d+ ms[^\n]*', newlog),
        'prompt_read': grab(r'prompt \d+ tokens = [^\n]*', newlog),
        'drafts': grab(r'drafts accepted \d+ of \d+', newlog),
    },
    'expert_cache': slots_line,
}
os.makedirs(f'{ROOT}/results', exist_ok=True)
with open(f'{ROOT}/results/{NAME}{SUFFIX}.json', 'w') as fh:
    json.dump(rec, fh, indent=1)
with open(f'{ROOT}/results/{NAME}{SUFFIX}.texts.json', 'w') as fh:
    json.dump({'greedy': [g['text'] for g in greedy], 'burst': [r['text'] for r in burst],
               'prefill': prefill['text']}, fh, indent=1)
print(json.dumps({'arm': NAME, 'greedy_tok_s': [g['wall_tok_s'] for g in greedy],
                  'burst_agg': rec['burst']['aggregate_tok_s'], 'prefill_tok_s': prefill['wall_tok_s'],
                  'prefill_prompt_tokens': prefill['prompt_tokens'],
                  'hashes': rec['parity'], 'cache': rec['expert_cache']}))
