#!/usr/bin/env python3
"""Long-prompt capture for the eviction-time V-flag test: one engine run on the 65K-token prompt with every dump on.

Dumps (all measurement-only, env-gated):
  STRATA_QSA_QDUMP + STRATA_QSA_QDUMP_PER_CHUNK=256 : prefill queries + outputs, last 256 rows of every chunk
  STRATA_QSA_DQDUMP                                 : decode-window queries (rotated space), every QSA layer
  STRATA_QSA_DUMP   + STRATA_QSA_DUMP_LAST=4096     : selected cells for the last 4096 prompt positions
Engine: dev build, served config args, --kv fp16 (the probe needs fp16 K), CPU share off, prompt only + 16 new tokens.
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dq_smoke as S  # noqa: E402  (config, exe, lib dirs)

OUT = S.ROOT / "tmp" / "kv-long"
OUT.mkdir(parents=True, exist_ok=True)
# argv[1]: rows per chunk for the query dump (256 = the last 256 rows; 8192 = every row of an 8192-token chunk)
QCHUNK = sys.argv[1] if len(sys.argv) > 1 else "256"
TAG = sys.argv[2] if len(sys.argv) > 2 else "longcode"
toks = (S.ROOT / "tmp" / "kv-realq" / "longcode.tokens.txt").read_text(encoding="utf-8").strip()
n_tok = len(toks.split(","))
tok_file = OUT / f"{TAG}.tokens.txt"
tok_file.write_text(toks, encoding="utf-8")

for name in (f"{TAG}.q.bin", f"{TAG}.dq.bin", f"{TAG}.sel.bin"):
    p = OUT / name
    if p.exists():
        p.unlink()

env = dict(os.environ)
env.update(S.CFG.get("env") or {})
env["STRATA_PREFILL_CPU_SHARE"] = "0"
env["STRATA_QSA_QDUMP"] = str(OUT / f"{TAG}.q.bin")
env["STRATA_QSA_QDUMP_PER_CHUNK"] = QCHUNK
env["STRATA_QSA_DQDUMP"] = str(OUT / f"{TAG}.dq.bin")
env["STRATA_QSA_DUMP"] = str(OUT / f"{TAG}.sel.bin")
env["STRATA_QSA_DUMP_LAST"] = "4096"
for d in S.LIB_DIRS:
    env["PATH"] = d + os.pathsep + env.get("PATH", "")

cmd = [S.DEV_EXE] + [str(a) for a in S.CFG["args"] if not isinstance(a, dict)] + [
    "--kv", "fp16", "--kv-resident", "0", "--max-context", str(n_tok + 64),
    "--tokens-file", str(tok_file), "--max-new", "16",
    "--dump-kv", str(OUT / f"{TAG}.kv.bin"),   # fp16 K/V of every QSA layer after prefill (the probe's cache)
]
print("prompt tokens", n_tok, flush=True)
with open(OUT / f"{TAG}.log", "w", encoding="utf-8") as lf:
    rc = subprocess.run(cmd, cwd=S.CFG["cwd"], env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=5400).returncode
print("engine rc", rc, flush=True)
for name in (f"{TAG}.q.bin", f"{TAG}.dq.bin", f"{TAG}.sel.bin"):
    p = OUT / name
    print(name, p.stat().st_size if p.exists() else "MISSING", flush=True)
sys.exit(rc)
