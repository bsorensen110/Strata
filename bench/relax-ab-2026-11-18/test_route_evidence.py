#!/usr/bin/env python3
"""Offline route-evidence classification checks."""
import json
import os
import tempfile

import gen_configs
from driver import cache_gate_failure
from route_evidence import (classify_route, has_accepted_wide_request, parse_expert_cache_slots,
                            iu8_route_status, parse_iu8_lines, read_dispatch_report,
                            validate_expert_cache_slots, validate_iu8_gate)


def report(path, launches, counts):
    with open(path, 'w') as fh:
        json.dump({'launches': launches, 'counts': counts}, fh)
    return read_dispatch_report(path)


def main():
    paths = gen_configs.assert_preflight()
    assert len(paths) == 6
    for name, arm, _ in gen_configs.arm_specs():
        cfg = json.load(open(f'{gen_configs.ROOT}/configs/{name}.json'))
        gen_configs.assert_cache_args(cfg['args'], arm, name)
        assert gen_configs.flag_value(cfg['args'], '--expert-cache', name) == '7500'
        assert arm['expert_cache_slots'] == 8736
        assert arm['expert_cache_budget_slots'] == 7500
    iu8_arm = next(arm for arm in json.load(open(
        f'{gen_configs.ROOT}/arms.json'))['arms'] if arm['id'] == 'iu8_relax')
    assert iu8_arm['switch'] == 'STRATA_HIP_MOE_IU8'
    assert iu8_arm['baseline'] == '0' and iu8_arm['arm'] == '1'
    assert iu8_arm['pin_arm'] == {'STRATA_HIP_RELAX_PACK': '1'}
    assert iu8_arm['treatment_only'] is True
    for role, expected_iu8, expected_relax in (('ctl', '0', None), ('trt', '1', '1')):
        cfg = json.load(open(f'{gen_configs.ROOT}/configs/{role}_iu8_relax.json'))
        assert cfg['port'] == (8300 if role == 'ctl' else 8301)
        assert cfg['env']['STRATA_HIP_MOE_IU8'] == expected_iu8
        assert cfg['env'].get('STRATA_HIP_RELAX_PACK') == expected_relax
        assert 'STRATA_HIP_MOE_X4' not in cfg['env']
        assert 'STRATA_HIP_DOT4_GEMV' not in cfg['env']

    cache_startup = ['strata generate: expert cache 8736 slots, 19.99 GiB of VRAM; policy is']
    assert parse_expert_cache_slots(cache_startup) == 8736
    assert validate_expert_cache_slots(cache_startup, 8736) == ('ok', 8736)
    assert validate_expert_cache_slots(cache_startup, 7500) == ('cache_mismatch', 8736)
    unrelated_cache = ['expert cache auto: deferred', 'elastic ExpertCache: 8736 slots']
    assert parse_expert_cache_slots(unrelated_cache) is None
    assert validate_expert_cache_slots(unrelated_cache, 8736) == ('cache_evidence_missing', None)
    assert validate_expert_cache_slots([], 8736) == ('cache_evidence_missing', None)
    mismatch = 'CACHE_GATE_FAIL kind=mismatch expected_actual=8736 actual=7500 budget=7500'
    missing = 'CACHE_GATE_FAIL kind=missing expected_actual=8736 actual=none budget=7500'
    assert cache_gate_failure(mismatch) == {
        'status': 'cache_mismatch', 'expected': 8736, 'actual': 7500,
        'budget': 7500, 'reason': mismatch,
    }
    assert cache_gate_failure(missing) == {
        'status': 'cache_evidence_missing', 'expected': 8736, 'actual': None,
        'budget': 7500, 'reason': missing,
    }

    suffix_positive = 'strata serve: suffix drafts: 5 windows, 15 of 54 drafts accepted'
    suffix_zero = 'strata serve: suffix drafts: 3 windows, 0 of 9 drafts accepted'
    lookup_positive = 'strata serve: lookup chain: 4 of 4 windows, 3 of 10 chained drafts accepted'
    assert has_accepted_wide_request([suffix_positive])
    assert not has_accepted_wide_request([suffix_zero])
    assert has_accepted_wide_request([lookup_positive])

    x4_t10_k0 = {
        'scope': [10, 0], 'cat': 'str',
        'key': 'HIP x4 GDN members=2: qkv|z (alpha/beta stay BF16)', 'n': 1,
    }
    x4_t8_k0 = {
        'scope': [8, 0], 'cat': 'str',
        'key': 'HIP x4 QSA members=2: qkv|z (alpha/beta stay BF16)', 'n': 1,
    }
    with tempfile.TemporaryDirectory() as tmp:
        report_path = os.path.join(tmp, 'synthetic.dispatch.json')
        # This T10 launch resembles startup warm-up. Generated-config preflight above proves
        # every arm disables that warm-up; attribution must not pretend launches are per-request.
        engaged = report(report_path, {'T10/k0': 42, 'T10/k1': 3, 'T8/k0': 99}, [x4_t10_k0])
        assert engaged['x4_captures']['n'] == 1
        assert [(entry['scope_t'], entry['kind'], entry['n'])
                for entry in engaged['x4_captures_by_scope']] == [(10, 0, 1)]
        assert engaged['x4_wide_scope_note_count'] == 1
        assert engaged['wide_launches'] == {'k0': 42, 'k1': 3, 'k2': 0}
        assert engaged['wide_launches_by_t']['T10'] == {'k0': 42, 'k1': 3, 'k2': 0}
        assert classify_route(True, engaged, [suffix_positive]) == (True, None)

        launches_only = report(
            os.path.join(tmp, 'launches-only.dispatch.json'), {'T10/k0': 42}, [])
        assert classify_route(True, launches_only, [lookup_positive]) == (
            False, 'route not engaged')

        wrong_scope = report(
            os.path.join(tmp, 'wrong-scope.dispatch.json'),
            {'T10/k0': 42}, [x4_t8_k0])
        assert classify_route(True, wrong_scope, [suffix_positive]) == (
            False, 'route not engaged')

        assert read_dispatch_report(os.path.join(tmp, 'missing.dispatch.json')) is None
        assert classify_route(True, None, [suffix_positive]) == (
            False, 'dispatch-report missing: evidence channel failure')
        assert classify_route(False, engaged, [suffix_positive]) == (
            False, 'X4 captured despite disabled')

        assert not has_accepted_wide_request([suffix_zero])
        assert classify_route(True, engaged, [suffix_zero]) == (
            False, 'route not engaged')

        malformed = os.path.join(tmp, 'malformed.dispatch.json')
        with open(malformed, 'w') as fh:
            json.dump({'launches': [], 'counts': []}, fh)
        try:
            read_dispatch_report(malformed)
        except ValueError as exc:
            assert str(exc) == 'invalid dispatch report shape'
        else:
            raise AssertionError('malformed report shape was accepted')

    accept_lines = [
        'HIP MoE IU8 layer 3: all 427 experts relaxed-W6 (worst block rel err 0.031)',
        'HIP MoE IU8 layer 4: all 427 experts relaxed-W6 (worst block rel err 0.028)',
        'strata serve: 4002 MiB of VRAM free with everything loaded',
    ]
    full = parse_iu8_lines(accept_lines)
    assert full == {
        'iu8_packed_layers': 2, 'iu8_refused_layers': 0, 'iu8_route_status': 'full',
        'vram_free_loaded_mib': 4002, 'iu8_alloc_error': None,
    }
    refused_lines = [
        'HIP MoE IU8 layer 3 refused: exactness proof failed; keeping incumbent',
        'HIP MoE IU8 layer 4 refused: exactness proof failed; keeping incumbent',
    ]
    refused = parse_iu8_lines(refused_lines)
    assert refused['iu8_packed_layers'] == 0
    assert refused['iu8_refused_layers'] == 2
    assert refused['iu8_route_status'] == 'absent'
    partial = parse_iu8_lines(accept_lines[:1] + refused_lines[:1])
    assert partial['iu8_packed_layers'] == 1
    assert partial['iu8_refused_layers'] == 1
    assert partial['iu8_route_status'] == 'partial'
    assert parse_iu8_lines([])['iu8_route_status'] == 'absent'
    assert iu8_route_status(refused, {'STRATA_HIP_MOE_IU8': '1'}) == 'absent'
    assert iu8_route_status(refused, {}) == 'off'
    assert iu8_route_status(full, {'STRATA_HIP_MOE_IU8': '1'}) == 'full'
    assert iu8_route_status(partial, {'STRATA_HIP_MOE_IU8': '1'}) == 'partial'
    assert validate_iu8_gate({'STRATA_HIP_MOE_IU8': '0'}, refused) == (True, None)
    assert validate_iu8_gate({}, refused) == (True, None)
    assert validate_iu8_gate({'STRATA_HIP_MOE_IU8': '1'}, refused) == (
        False, 'iu8 route not engaged (packed=0 refused=2)')
    assert validate_iu8_gate({'STRATA_HIP_MOE_IU8': '1'}, full) == (True, None)
    errors = parse_iu8_lines([
        'hipMalloc failed while allocating scratch',
        'out of memory while loading another buffer',
    ])
    assert errors['iu8_alloc_error'] == 'hipMalloc failed while allocating scratch'
    assert validate_iu8_gate({'STRATA_HIP_MOE_IU8': '1'}, errors) == (
        False, 'iu8 allocation error: hipMalloc failed while allocating scratch')


if __name__ == '__main__':
    main()
