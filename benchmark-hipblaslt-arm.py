#!/usr/bin/env python3
"""Run one isolated baseline/tuned arm, preserving results and stopping our server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import psutil
import requests

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / '.venv/bin/python')
BASE = 'http://127.0.0.1:8086'
parser = argparse.ArgumentParser()
parser.add_argument('--arm', choices=('baseline', 'tuned'), required=True)
parser.add_argument('--config', type=Path, required=True)
args = parser.parse_args()
DIRECTORY = ROOT / 'hipblaslt-benchmark' / args.arm
DIRECTORY.mkdir(parents=True, exist_ok=True)
config = json.loads(args.config.read_text())
canonical_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
config['log'] = str(DIRECTORY / 'engine.log')
config.setdefault('env', {})['STRATA_HIPBLASLT_VERBOSE'] = '1'
config_path = DIRECTORY / 'config.json'
config_path.write_text(json.dumps(config, indent=2) + '\n')
result = dict(arm=args.arm, original_config=str(args.config),
              original_config_sha256=canonical_hash, configuration=config,
              engine_sha256=hashlib.sha256((ROOT / 'engine/strata').read_bytes()).hexdigest(),
              started_epoch=time.time(), passed=False)
report = DIRECTORY / 'arm-results.json'

def save():
    report.write_text(json.dumps(result, indent=2) + '\n')

def engine_pids():
    return sorted(p.pid for p in psutil.process_iter(['exe'])
                  if p.info['exe'] == str(ROOT / 'engine/strata'))

def run(command, name, timeout=900):
    with (DIRECTORY / f'{name}-console.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)

assert not engine_pids(), 'A Strata engine is already running; do not interfere with it'
with socket.socket() as sock:
    assert sock.connect_ex(('127.0.0.1', 8086)) != 0, 'Port 8086 is already occupied'
server = None
owned_engines = []
try:
    with (DIRECTORY / 'server-console.log').open('w') as log:
        server = subprocess.Popen([PYTHON, '-u', '-m', 'serve.server', '--engine', 'strata',
                                   '--config', str(config_path), '--host', '127.0.0.1',
                                   '--port', '8086'], cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    result['server_pid'] = server.pid
    save()
    deadline = time.monotonic() + 180
    while True:
        if server.poll() is not None:
            raise RuntimeError(f'Server exited at startup: {server.returncode}')
        try:
            response = requests.get(BASE + '/health', timeout=3)
            response.raise_for_status()
            result['health'] = response.json()
            break
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise RuntimeError('Server readiness deadline exceeded')
            time.sleep(1)
    result['engine_pids_before'] = engine_pids()
    assert len(result['engine_pids_before']) == 1
    owned_engines = [psutil.Process(pid) for pid in result['engine_pids_before']]
    print(f'{args.arm}: server ready, engine={result["engine_pids_before"]}', flush=True)
    run([PYTHON, '-u', 'benchmark-hipblaslt-prefill.py', '--url', BASE,
         '--model', config['model_name'], '--engine-log', config['log'],
         '--output', str(DIRECTORY / 'prefill-results.json'), '--label', args.arm], 'prefill')
    prefill = json.loads((DIRECTORY / 'prefill-results.json').read_text())
    fresh = [row for row in prefill if row['kind'] == 'fresh']
    assert len(prefill) == 15 and len(fresh) == 6
    assert all(row['metrics']['reused'] == 0 for row in fresh)
    assert all(row['message'].get('content') and row['finish_reason'] != 'error' for row in prefill)
    assert engine_pids() == result['engine_pids_before'], 'Engine restarted during prefill benchmark'
    result['prefill_trials'] = len(fresh)
    save()
    print(f'{args.arm}: six fresh long-prompt trials passed', flush=True)
    run([PYTHON, '-u', 'benchmark-hipblaslt-decode.py', '--config', str(config_path),
         '--output', str(DIRECTORY / 'decode-results.json')], 'decode')
    decode = json.loads((DIRECTORY / 'decode-results.json').read_text())
    assert decode['passed'] and len(decode['trials']) == 3
    result['decode_summary'] = decode['summary']
    assert engine_pids() == result['engine_pids_before'], 'Engine restarted during decode benchmark'
    if args.arm == 'tuned':
        run([PYTHON, '-u', 'verify-install-hipblaslt.py'], 'verification', timeout=300)
        verification = json.loads((ROOT / 'verification-hipblaslt-results.json').read_text())
        assert verification['passed']
        result['verification'] = verification
    result['engine_pids_after'] = engine_pids()
    assert result['engine_pids_after'] == result['engine_pids_before']
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == canonical_hash
    result['passed'] = True
except Exception as error:
    result['error'] = f'{type(error).__name__}: {error}'
    print(result['error'], flush=True)
finally:
    if server is not None and server.poll() is None:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
    for process in owned_engines:
        if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            process.terminate()
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    result['remaining_engine_pids'] = engine_pids()
    result['server_returncode'] = server.returncode if server else None
    result['finished_epoch'] = time.time()
    result['engine_diagnostics'] = [line for line in Path(config['log']).read_text(errors='replace').splitlines()
                                    if 'prefill gemm:' in line] if Path(config['log']).exists() else []
    if result['remaining_engine_pids']:
        result['passed'] = False
        result['error'] = 'Engine remained running after benchmark shutdown'
    save()
print(json.dumps({key: result.get(key) for key in ('arm', 'passed', 'error', 'decode_summary',
                                                'engine_pids_before', 'engine_pids_after',
                                                'remaining_engine_pids')}, indent=2), flush=True)
raise SystemExit(0 if result['passed'] else 1)
