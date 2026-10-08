#!/usr/bin/env python3
"""Compile the actual BF16 calibration/parity encoders as host C++ and test them."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
scratch = Path(os.environ['TMPDIR'])
for label, source, function in (
    ('calibrator', 'tools/hip/tune_hipblaslt.cpp', 'input_bits'),
    ('parity', 'tests/hip/prefill_hipblaslt_gemm.cpp', 'encode'),
):
    output = scratch / f'strata-bf16-{label}-regression'
    command = [
        'g++', '-std=c++20', '-O3', '-ffunction-sections', '-fdata-sections',
        '-D__HIP_PLATFORM_AMD__=1', '-DSTRATA_USE_HIP=1',
        f'-DBF16_ENCODER_SOURCE="{root / source}"', f'-DBF16_ENCODER_FUNCTION={function}',
        '-Iinclude', '-Iinclude/strata/hip_compat', '-Isrc/prefill', '-I/opt/rocm/include',
        'tests/hip/bf16_host_encoding_regression.cpp', '-Wl,--gc-sections',
        '-L/opt/rocm/lib', '-Wl,-rpath,/opt/rocm/lib', '-lhipblaslt', '-lhipblas',
        '-lamdhip64', '-o', str(output),
    ]
    subprocess.run(command, cwd=root, check=True, timeout=90)
    subprocess.run([str(output)], cwd=root, check=True, timeout=30)
    print(f'{label}: PASS', flush=True)
