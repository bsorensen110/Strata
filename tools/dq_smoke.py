#!/usr/bin/env python3
"""Smoke test for STRATA_QSA_DQDUMP: run the DEV engine on a short prompt, once with the dump and once without.

Checks (printed, no pass/fail hidden):
  1. with the dump: the .dq.bin file has records {qsa layer, pos0, T, NH*HD} and payload sizes T*NH*HD*4
  2. without the dump: no file is written and the generated tokens are identical to the dumped run (neutrality)
Uses the served config's args (read from the JSON, not retyped) with the dev exe, CUDA 13.0 lib dirs, --kv fp16.
Service must be stopped (the GPU holds one engine).
"""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "strata-swift-iq3_xxs.json").read_text(encoding="utf-8"))
DEV_EXE = str(ROOT / "build-cu130" / "strata.exe")
LIB_DIRS = [r"C:\Users\blsor\cuda130\nvidia\cu13\bin", r"C:\Users\blsor\cuda130\nvidia\cu13\bin\x64"]
OUT = ROOT / "tmp" / "kv-realq"
TOKENS = (OUT / "code.tokens.txt").read_text(encoding="utf-8").strip().split(",")
PROMPT = [int(t) for t in TOKENS[:600]]


def run(tag: str, env_extra: dict, n_new: int) -> tuple[int, list[int]]:
    tok_file = OUT / f"smoke-{tag}.tokens.txt"
    tok_file.write_text(",".join(map(str, PROMPT)), encoding="utf-8")
    env = dict(os.environ)
    env.update(CFG.get("env") or {})
    env.update(env_extra)
    # Prompt chunks under 1024 tokens share MoE experts with the idle CPU, and that split varies run to run,
    # so greedy tokens are not reproducible without this (see the strata-kv-kernel-work skill).
    env["STRATA_PREFILL_CPU_SHARE"] = "0"
    for d in LIB_DIRS:
        env["PATH"] = d + os.pathsep + env.get("PATH", "")
    cmd = [DEV_EXE] + [str(a) for a in CFG["args"] if not isinstance(a, dict)] + [
        "--kv", "fp16", "--kv-resident", "0", "--max-context", "8192",
        "--tokens-file", str(tok_file), "--max-new", str(n_new),
    ]
    log = OUT / f"smoke-{tag}.log"
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, cwd=CFG["cwd"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=1500)
    text = p.stdout.decode("utf-8", errors="replace")
    log.write_text(text, encoding="utf-8")
    # the engine prints the generated ids on a line starting "output  :", space-separated
    gen = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("output") and ":" in s:
            gen = [int(x) for x in s.split(":", 1)[1].split() if x.lstrip("-").isdigit()]
    return p.returncode, gen


def check_records(path: Path, nh_hd: int, n_layers: int) -> None:
    data = path.read_bytes()
    off, recs = 0, []
    while off < len(data):
        q, pos0, T, zv = struct.unpack_from("<4i", data, off)
        off += 16
        nbytes = T * zv * 4
        recs.append((q, pos0, T, zv, nbytes))
        off += nbytes
    print("records:", len(recs), "| layers seen:", sorted({r[0] for r in recs}))
    bad = [r for r in recs if r[3] != nh_hd or r[4] != r[2] * nh_hd * 4]
    print("size check (zv == NH*HD, payload == T*zv*4):", "OK" if not bad else f"BAD {bad[:3]}")
    print("layer count per window:", len(recs) // max(1, len({(r[1], r[2]) for r in recs})),
          "(expect", n_layers, ")")


if __name__ == "__main__":
    dq = OUT / "smoke-dq.dq.bin"
    if dq.exists():
        dq.unlink()
    rc1, gen1 = run("dq", {"STRATA_QSA_DQDUMP": str(dq)}, 16)
    print("dumped run rc", rc1, "| generated", len(gen1), "tokens")
    rc2, gen2 = run("plain", {}, 16)
    print("plain run rc", rc2, "| generated", len(gen2), "tokens")
    print("neutrality (identical generated tokens):", gen1 == gen2 and len(gen1) > 0)
    if dq.exists():
        check_records(dq, 24 * 256, 12)
    else:
        print("NO dump file written")
    sys.exit(0 if rc1 == 0 and rc2 == 0 else 1)
