#!/usr/bin/env python3
"""Long-context trial for one campaign arm: 64k and 128k token prompts.

For each target size:
  1. build a prompt of about that many tokens and calibrate it against the engine's own
     usage.prompt_tokens with a 1-token request (the time that request takes IS the prompt
     processing time: TTFT including prefill, since the reply is a single token);
  2. send the same prompt again for 64 tokens -- the conversation cache holds the prompt, so
     this measures decode at that context depth (warm), and reports whether the cache hit.

Writes results/<arm>_long.json. Readiness/verification mirrors bench_arm.py.
"""
import hashlib, json, os, subprocess, sys, time, urllib.request

ROOT = '/home/soren/Strata-latest-20261112-182859/bench/ports-ab-2026-11-15-ctx'
NAME = sys.argv[1]
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ''
TARGETS = [(65536, '64k'), (131072, '128k')]
cfg = json.load(open(f'{ROOT}/configs/{NAME}.json'))
port = cfg['port']
url = f'http://127.0.0.1:{port}'
unit = f'portsab-{NAME}'

key = None
for line in open('/home/soren/.config/strata-upstream-v0.2.10/production.env'):
    line = line.strip()
    if line.startswith('STRATA_API_KEY') and '=' in line:
        key = line.split('=', 1)[1].strip().strip('"').strip("'")
if not key:
    sys.exit('no api key found')

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
    alive = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip()
    if alive != 'active':
        sys.exit(f'{NAME}: engine not running before the long-context trials (state={alive})')
    time.sleep(3)
if model is None:
    sys.exit(f'{NAME}: readiness timeout')

# one paragraph, repeated; the engine's tokenizer decides the real count, so calibrate per target
PARA = ('The quick brown fox jumps over the lazy dog while a radix tree stores prefixes, '
        'counting every edge with a small table and a sorted list of children. ')


def post(prompt, max_tokens, temp=0.0, timeout=1800):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': temp,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + str(key), 'Content-Type': 'application/json'})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    dt = time.monotonic() - t
    u = d.get('usage') or {}
    ch = d['choices'][0]
    txt = ch['message'].get('content') or ''
    return {'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': u.get('completion_tokens'),
            'cached_tokens': (u.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'finish': ch.get('finish_reason'), 'wall_s': round(dt, 3), 'text': txt,
            'hash': hashlib.sha256(txt.encode()).hexdigest()}


out = {'arm': NAME, 'round': SUFFIX or 'r1', 'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
       'exe': cfg['exe'], 'binary_sha256': hashlib.sha256(open(cfg['exe'], 'rb').read()).hexdigest(),
       'arm_env': cfg.get('env', {}), 'resident_budget_arg': [a for a in cfg['args'] if 'resident' in a or 'arena' in a],
       'contexts': []}

for target, label in TARGETS:
    # calibrate the repetition count: one probe, then scale
    reps = max(16, target // 40)
    got = None
    attempt = 0
    prompt = ''
    probe = {'wall_s': 0.0, 'prompt_tokens': None, 'cached_tokens': None}
    for attempt in range(4):
        # the same prompt is reused for the deep trial, so the conversation cache still hits; the closing
        # instruction asks for a long answer so the deep trial measures decode, not a two-token "ok."
        prompt = (f'[ports-ab {label} ctx] ' + PARA * reps +
                  '\nWrite a detailed 200-word explanation of how a radix tree stores and looks up prefixes.')
        probe = post(prompt, 1)
        got = probe['prompt_tokens']
        if got is None:
            sys.exit(f'{NAME}: no usage in the {label} probe')
        if abs(got - target) / target <= 0.04:
            break
        reps = max(8, int(reps * target / got))
    ttft = probe['wall_s']
    prompt_tok_s = round(got / ttft, 1) if ttft > 0 else None
    deep = post(prompt, 64)
    decode_s = deep['wall_s']
    rec = {'label': label, 'target': target, 'prompt_tokens': got,
           'prompt_wall_s': ttft, 'prompt_tokens_per_s': prompt_tok_s,
           'cached_tokens_probe': probe.get('cached_tokens'),
           'decode64': {k: deep[k] for k in ('completion_tokens', 'finish', 'wall_s', 'hash', 'cached_tokens')},
           'decode_tok_s': round(deep['completion_tokens'] / decode_s, 2) if decode_s > 0 else None,
           'calibration_attempts': attempt + 1}
    out['contexts'].append(rec)
    print(f'{NAME} {label}: {got} tok prompt, prefill {ttft:.1f}s ({prompt_tok_s} tok/s), '
          f'decode {rec["decode_tok_s"]} tok/s, cached {deep.get("cached_tokens")}', flush=True)

with open(f'{ROOT}/results/{NAME}{SUFFIX}_long.json', 'w') as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({'arm': NAME, 'long': [{c['label']: [c['prompt_tokens'], c['prompt_tokens_per_s'], c['decode_tok_s']] for c in out['contexts']}]}))
