#!/usr/bin/env python3
"""Parse dispatch reports and classify route evidence without requiring an engine."""
import json
import os
import re


def parse_dispatch_report(data):
    if not isinstance(data, dict) or not isinstance(data.get('launches'), dict) or \
            not isinstance(data.get('counts'), list):
        raise ValueError('invalid dispatch report shape')
    captures = []
    for item in data['counts']:
        if not isinstance(item, dict):
            continue
        key = item.get('key')
        if isinstance(key, str) and (key.startswith('HIP x4 GDN members') or
                                     key.startswith('HIP x4 QSA members')):
            scope = item.get('scope') or []
            captures.append({
                'scope_t': scope[0] if len(scope) > 0 else None,
                'kind': scope[1] if len(scope) > 1 else None,
                'key': key,
                'n': int(item.get('n') or 0),
            })
    by_width = {f'T{token_count}': {f'k{k}': 0 for k in range(3)}
                for token_count in range(9, 65)}
    for token_count in range(9, 65):
        for kind in range(3):
            by_width[f'T{token_count}'][f'k{kind}'] = int(
                data['launches'].get(f'T{token_count}/k{kind}', 0) or 0)
    wide = {f'k{k}': sum(by_width[f'T{token_count}'][f'k{k}']
                         for token_count in range(9, 65)) for k in range(3)}
    wide_scope_captures = [
        item for item in captures
        if type(item['scope_t']) is int and item['scope_t'] >= 9 and
        type(item['kind']) is int and item['kind'] in (0, 1) and item['n'] > 0
    ]
    return {
        'x4_captures': {'n': sum(item['n'] for item in captures), 'entries': captures},
        'x4_captures_by_scope': captures,
        'x4_wide_scope_note_count': sum(item['n'] for item in wide_scope_captures),
        'wide_scope_captures': wide_scope_captures,
        'wide_launches': wide,
        'wide_launches_by_t': by_width,
        'launches': data['launches'],
        'counts_preview': data['counts'][:20],
    }


def read_dispatch_report(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return parse_dispatch_report(json.load(fh))


def parse_expert_cache_slots(lines):
        """Return actual slots from an engine startup line, or None if none is present."""
        for line in lines:
            match = re.search(r'\bexpert cache\s+(\d+)\s+slots\b', line, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None


def validate_expert_cache_slots(lines, expected):
        """Return (status, actual), distinguishing absent evidence from a mismatch."""
        actual = parse_expert_cache_slots(lines)
        if actual is None:
            return 'cache_evidence_missing', None
        if actual != expected:
            return 'cache_mismatch', actual
        return 'ok', actual


def parse_iu8_lines(lines):
    """Parse IU8 route and memory evidence from a complete launch log."""
    packed = 0
    refused = 0
    vram_free = None
    alloc_error = None
    for line in lines:
        if re.search(r'HIP MoE IU8 layer \d+: all \d+ experts relaxed-W6\b', line):
            packed += 1
        if re.search(r'HIP MoE IU8 layer \d+ refused\b', line):
            refused += 1
        vram_match = re.search(
            r'strata serve: (\d+) MiB of VRAM free with everything loaded', line)
        if vram_match:
            vram_free = int(vram_match.group(1))
        if alloc_error is None and re.search(
                r'hipMalloc|out of memory|allocation failed', line, re.IGNORECASE):
            alloc_error = line.rstrip('\r\n')
    return {
        'iu8_packed_layers': packed,
        'iu8_refused_layers': refused,
        'iu8_route_status': ('full' if packed and not refused else
                             'partial' if packed and refused else 'absent'),
        'vram_free_loaded_mib': vram_free,
        'iu8_alloc_error': alloc_error,
    }


def validate_iu8_gate(arm_env, evidence):
    """Require an engaged, allocation-clean IU8 route only when explicitly enabled."""
    if arm_env.get('STRATA_HIP_MOE_IU8') != '1':
        return True, None
    if evidence.get('iu8_alloc_error') is not None:
        return False, f'iu8 allocation error: {evidence["iu8_alloc_error"]}'
    packed = evidence.get('iu8_packed_layers', 0)
    if packed <= 0:
        refused = evidence.get('iu8_refused_layers', 0)
        return False, f'iu8 route not engaged (packed=0 refused={refused})'
    return True, None


def iu8_route_status(evidence, arm_env):
    packed = evidence['iu8_packed_layers']
    refused = evidence['iu8_refused_layers']
    if packed:
        return 'partial' if refused else 'full'
    return 'off' if arm_env.get('STRATA_HIP_MOE_IU8') != '1' else 'absent'


def has_accepted_wide_request(lines):
    for line in lines:
        suffix = re.search(
            r'suffix drafts:\s*(\d+) windows,\s*(\d+) of\s*(\d+) drafts accepted',
            line,
            re.IGNORECASE,
        )
        if suffix and int(suffix.group(2)) > 0:
            return True
        lower = line.lower()
        if 'lookup chain:' in lower and 'chained drafts accepted' in lower:
            try:
                accepted = int(lower.split('chained drafts accepted', 1)[0].rsplit(',', 1)[1]
                               .split('of', 1)[0].strip())
            except (IndexError, ValueError):
                continue
            if accepted > 0:
                return True
    return False


def classify_route(route_expected, evidence, wide_request_lines):
    """Require request-era draft acceptance plus a scoped X4 note and matching launch."""
    if evidence is None:
        return False, 'dispatch-report missing: evidence channel failure'
    x4_count = evidence['x4_captures']['n']
    if not route_expected and x4_count > 0:
        return False, 'X4 captured despite disabled'
    if route_expected:
        matching_scope = any(
            item['n'] > 0 and item['scope_t'] >= 9 and item['kind'] in (0, 1) and
            evidence['wide_launches_by_t'].get(f"T{item['scope_t']}", {}).get(
                f"k{item['kind']}", 0) > 0
            for item in evidence['x4_captures']['entries']
            if type(item['scope_t']) is int and type(item['kind']) is int
        )
        if not matching_scope or not has_accepted_wide_request(wide_request_lines):
            return False, 'route not engaged'
    return True, None
