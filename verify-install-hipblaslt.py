#!/usr/bin/env python3
"""Real-model API smoke tests, not a throughput or full-window benchmark."""
import json
import re
import time
from pathlib import Path
import psutil
import requests

BASE = 'http://127.0.0.1:8086'
OUT = Path(__file__).with_name('verification-hipblaslt-results.json')

def engine_pids():
    result = []
    for process in psutil.process_iter(['pid', 'exe']):
        try:
            if process.info['exe'] == '/home/soren/Strata/engine/strata':
                result.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return sorted(result)

def get(path):
    response = requests.get(BASE + path, timeout=15)
    response.raise_for_status()
    return response.json()

def chat(model, messages, max_tokens=96):
    started = time.monotonic()
    response = requests.post(BASE + '/v1/chat/completions', json={
        'model': model,
        'messages': messages,
        'stream': False,
        'temperature': 0,
        'max_tokens': max_tokens,
        'reasoning_effort': 'none',
    }, timeout=180)
    response.raise_for_status()
    data = response.json()
    data['wall_seconds'] = round(time.monotonic() - started, 3)
    return data

result = {'checks': [], 'started_epoch': time.time()}
try:
    result['health'] = get('/health')
    result['models'] = get('/v1/models')
    model = result['models']['data'][0]['id']
    result['engine_pids_before'] = engine_pids()
    assert result['engine_pids_before'], 'No real Strata engine is running'
    expected = str(17 * 19)
    response = chat(model, [{'role': 'user', 'content': 'Calculate 17 times 19. Reply with only the integer answer.'}])
    content = response['choices'][0]['message']['content'] or ''
    assert content.strip() == expected, f'Incorrect arithmetic: {content!r}'
    assert response['choices'][0]['finish_reason'] != 'error', response
    result['checks'].append({'name': 'arithmetic', 'passed': True, 'response': response})
    response = chat(model, [
        {'role': 'system', 'content': 'Remember this exact marker: strata-r9700-ready. When asked for the marker, reply with only the marker.'},
        {'role': 'user', 'content': 'What is the marker?'},
    ])
    content = response['choices'][0]['message']['content'] or ''
    assert content.strip() == 'strata-r9700-ready', f'Incorrect marker: {content!r}'
    result['checks'].append({'name': 'system_marker_recall', 'passed': True, 'response': response})
    messages = [{'role': 'user', 'content': 'Our project codename is Copper Heron. Acknowledge in one word.'}]
    first = chat(model, messages, 32)
    content = first['choices'][0]['message']['content'] or ''
    assert content.strip(), 'Empty acknowledgement'
    messages.append({'role': 'assistant', 'content': content})
    messages.append({'role': 'user', 'content': 'What is our project codename? Reply with only its two words.'})
    response = chat(model, messages, 32)
    content = response['choices'][0]['message']['content'] or ''
    assert 'Copper Heron' in content, f'Incorrect conversation recall: {content!r}'
    result['checks'].append({'name': 'multi_turn_recall', 'passed': True, 'first_response': first, 'response': response})
    response = requests.post(BASE + '/v1/chat/completions', json={
        'model': model, 'messages': [{'role': 'user', 'content': 'Reply with only the word READY.'}],
        'stream': True, 'temperature': 0, 'max_tokens': 32, 'reasoning_effort': 'none',
    }, stream=True, timeout=180)
    response.raise_for_status()
    chunks = []
    text = ''
    done = False
    for raw_line in response.iter_lines():
        line = raw_line.decode('utf-8') if isinstance(raw_line, bytes) else raw_line
        if not line or not line.startswith('data: '):
            continue
        payload = line[6:]
        if payload == '[DONE]':
            done = True
            break
        data = json.loads(payload)
        chunks.append(data)
        for choice in data.get('choices', []):
            text += choice.get('delta', {}).get('content') or ''
    response.close()
    assert done and text.strip() == 'READY', f'Incorrect stream: done={done}, text={text!r}'
    result['checks'].append({'name': 'streaming', 'passed': True, 'text': text, 'chunks': chunks})
    result['engine_pids_after'] = engine_pids()
    assert result['engine_pids_after'] == result['engine_pids_before'], 'Engine restarted during tests'
    result['final_health'] = get('/health')
    result['status'] = get('/status')
    result['passed'] = True
except Exception as error:
    result['passed'] = False
    result['error'] = f'{type(error).__name__}: {error}'
finally:
    result['finished_epoch'] = time.time()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
raise SystemExit(0 if result['passed'] else 1)
