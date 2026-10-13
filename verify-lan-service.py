#!/usr/bin/env python3
"""Probe the configured LAN listener and real model; not a remote-client test."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'lan-service-verification.json')
parser.add_argument('--wait-seconds', type=float, default=150)
args = parser.parse_args()
config = json.loads((ROOT / 'strata-swift-iq3_xxs.json').read_text())
base = f"http://{config['host']}:8086"
session = requests.Session()
session.trust_env = False
result = {'base_url': base, 'remote_client_verified': False, 'checks': [], 'passed': False}

def properties(command):
    text = subprocess.check_output(command, text=True)
    return dict(line.split('=', 1) for line in text.splitlines())

def service():
    return properties(['systemctl', '--user', 'show', 'strata.service', '-p', 'ActiveState',
                       '-p', 'SubState', '-p', 'MainPID', '-p', 'NRestarts', '-p', 'UnitFileState'])

try:
    result['service_before'] = service()
    assert result['service_before']['ActiveState'] == 'active', result['service_before']
    assert result['service_before']['UnitFileState'] == 'enabled', result['service_before']
    result['login'] = properties(['loginctl', 'show-user', 'soren', '-p', 'Linger'])
    assert result['login']['Linger'] == 'yes', result['login']
    deadline = time.monotonic() + args.wait_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = session.get(base + '/health', timeout=5)
            response.raise_for_status()
            result['health'] = response.json()
            break
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)
            time.sleep(1)
    else:
        raise RuntimeError(f'LAN health deadline exceeded: {last_error}')
    result['checks'].append('LAN health HTTP 200')
    response = session.get(base + '/', timeout=10)
    response.raise_for_status()
    assert 'text/html' in response.headers.get('Content-Type', '')
    assert '<html' in response.text.lower()
    result['checks'].append('LAN web UI HTTP 200 / HTML')
    response = session.get(base + '/v1/models', timeout=10)
    response.raise_for_status()
    result['models'] = response.json()
    model = result['models']['data'][0]['id']
    result['checks'].append('LAN OpenAI model listing HTTP 200')
    response = session.post(base + '/v1/chat/completions', json={
        'model': model,
        'messages': [{'role': 'user', 'content': 'Calculate 17 times 19. Reply with only the integer answer.'}],
        'stream': False, 'temperature': 0, 'max_tokens': 96, 'reasoning_effort': 'none',
    }, timeout=180)
    response.raise_for_status()
    result['inference'] = response.json()
    choice = result['inference']['choices'][0]
    assert (choice['message']['content'] or '').strip() == str(17 * 19), choice
    assert choice['finish_reason'] != 'error', choice
    result['checks'].append('LAN real-model arithmetic passed')
    result['service_after'] = service()
    assert result['service_after'] == result['service_before'], result['service_after']
    result['checks'].append('Stable systemd process, enabled, linger=yes')
    result['passed'] = True
except Exception as error:
    result['error'] = f'{type(error).__name__}: {error}'
finally:
    session.close()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
raise SystemExit(0 if result['passed'] else 1)
