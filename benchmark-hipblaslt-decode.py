#!/usr/bin/env python3
"""Three sustained real-model decode trials after warmup; unchanged run config."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
OUT = args.output
BASE = 'http://127.0.0.1:8086'
CONFIG = args.config
PROMPTS = [
    ('storage', 'CASE A: Write a thorough practical guide of at least 2000 words to designing reliable home-server backups. Cover threat models, snapshots versus backups, the 3-2-1 rule, retention, encryption, offline copies, restore drills, monitoring, and a concrete example schedule. Use descriptive sections and explain tradeoffs. Start directly with the guide and continue in detail; do not produce a short summary.'),
    ('networking', 'CASE B: Write a thorough practical guide of at least 2000 words to diagnosing intermittent network latency in a Linux home lab. Explain baseline measurements, ping limitations, packet loss, DNS, MTU, bufferbloat, Ethernet errors, Wi-Fi interference, routing, and server load. Include commands and explain how to interpret results. Start directly with the guide and continue in detail; do not produce a short summary.'),
    ('software', 'CASE C: Write a thorough practical guide of at least 2000 words to building a maintainable Python web API. Explain module boundaries, validation, error handling, persistence, transactions, tests, observability, configuration, deployment, and backward compatibility. Include realistic examples and tradeoffs. Start directly with the guide and continue in detail; do not produce a short summary.'),
]

def pids():
    return sorted(p.pid for p in psutil.process_iter(['exe'])
                  if p.info['exe'] == str(ROOT / 'engine/strata'))

result = {
    'model': 'swift-1.5-iq3_xxs',
    'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    'configuration': json.loads(CONFIG.read_text()),
    'sampling': {'temperature': 0, 'top_k': 1, 'top_p': 1, 'min_p': 0, 'seed': 42, 'reasoning_effort': 'none'},
    'output_cap': 512,
    'trials': [],
    'started_epoch': time.time(),
    'notes': ['Warm model/filesystem state; configuration unchanged within this arm.',
              'Output is intentionally capped at 512 tokens; this is throughput measurement, not completed-task quality.',
              'Decode throughput uses engine timing, not HTTP wall time.'],
}

def save():
    OUT.write_text(json.dumps(result, indent=2) + '\n')

def request(label, prompt, cap):
    started = time.monotonic()
    response = requests.post(BASE + '/v1/chat/completions', json={
        'model': result['model'],
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'max_tokens': cap,
        **result['sampling'],
    }, timeout=240)
    response.raise_for_status()
    data = response.json()
    seconds = time.monotonic() - started
    choice = data['choices'][0]
    assert choice['message'].get('content'), f'{label}: empty output'
    assert choice.get('finish_reason') != 'error', data
    row = {'label': label, 'prompt': prompt, 'wall_seconds': seconds,
           'usage': data['usage'], 'timings': data['timings'],
           'finish_reason': choice['finish_reason'], 'message': choice['message'], 'response': data}
    print(json.dumps({k: row[k] for k in ('label', 'wall_seconds', 'usage', 'timings', 'finish_reason')}), flush=True)
    return row

try:
    deadline = time.monotonic() + 150
    while True:
        try:
            response = requests.get(BASE + '/health', timeout=3)
            response.raise_for_status()
            result['health'] = response.json()
            break
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise RuntimeError('Strata did not become ready within 150 seconds')
            time.sleep(2)
    result['engine_pids_before'] = pids()
    assert len(result['engine_pids_before']) == 1, result['engine_pids_before']
    result['warmup'] = request('warmup', 'Explain what a Linux process is and describe its lifecycle in approximately 200 words.', 128)
    save()
    for label, prompt in PROMPTS:
        row = request(label, prompt, 512)
        result['trials'].append(row)
        save()
        assert row['usage']['completion_tokens'] == 512, f'{label}: did not generate the requested benchmark length'
        assert row['finish_reason'] == 'length', f'{label}: unexpected finish reason'
        assert row['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0, f'{label}: reused cached prompt tokens'
        assert pids() == result['engine_pids_before'], 'Engine restarted during benchmark'
    rows = result['trials']
    assert len(rows) == 3 and len({r['label'] for r in rows}) == 3
    tokens = sum(r['usage']['completion_tokens'] for r in rows)
    decode_ms = sum(r['timings']['predicted_ms'] for r in rows)
    rates = [r['timings']['predicted_per_second'] for r in rows]
    result['summary'] = {
        'trials': len(rows), 'generated_tokens': tokens,
        'decode_seconds': decode_ms / 1000,
        'weighted_decode_tokens_per_second': tokens / (decode_ms / 1000),
        'median_decode_tokens_per_second': statistics.median(rates),
        'min_decode_tokens_per_second': min(rates),
        'max_decode_tokens_per_second': max(rates),
        'total_wall_seconds': sum(r['wall_seconds'] for r in rows),
    }
    result['engine_pids_after'] = pids()
    result['passed'] = True
    print('SUMMARY ' + json.dumps(result['summary']), flush=True)
except Exception as error:
    result['passed'] = False
    result['error'] = f'{type(error).__name__}: {error}'
    print('ERROR ' + result['error'], flush=True)
finally:
    result['finished_epoch'] = time.time()
    save()
raise SystemExit(0 if result.get('passed') else 1)
