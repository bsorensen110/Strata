#!/usr/bin/env python3
"""Greedy A/B bench for one running Strata candidate.

usage: bench.py <cand.json> <out.json> <label>
Does: arithmetic smoke, warm-up, single-request runs, 4-way concurrent runs.
Temperature 0 throughout so outputs are comparable across arms. API key is read
from the candidate config and never printed or written to the output file.
"""
import concurrent.futures as cf
import hashlib
import json
import statistics
import sys
import time
import urllib.request

cfg_path, out_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(cfg_path))
PORT = cfg["port"]
MODEL = cfg["model_name"]
KEY = cfg["api_key"]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PROMPTS = [
    "Explain how a hash table handles collisions, in about 250 words.",
    "Write a Python function that merges two sorted lists, with comments.",
    "Describe the water cycle for a 10-year-old in a short list of steps.",
    "Summarize the causes of the French Revolution in six bullet points.",
]
MAX_TOK = 384


def call(prompt, max_tokens=MAX_TOK):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ch = d["choices"][0]
    usage = d.get("usage", {})
    return {
        "text": ch["message"].get("content") or "",
        "finish": ch.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed_s": dt,
    }


res = {"label": label, "port": PORT, "model": MODEL, "max_tokens": MAX_TOK,
       "temperature": 0}

smoke = call("What is 6*7? Answer with just the number.", max_tokens=256)
res["smoke"] = {"text": smoke["text"], "finish": smoke["finish"]}
print(f"[{label}] smoke: {smoke['text'][:40]!r} finish={smoke['finish']}", flush=True)

call(PROMPTS[0])  # warm-up, excluded from timing
print(f"[{label}] warm-up done", flush=True)

singles = []
for i in range(2):
    r = call(PROMPTS[0])
    singles.append({"tok": r["completion_tokens"], "elapsed_s": r["elapsed_s"],
                    "tok_s": r["completion_tokens"] / r["elapsed_s"],
                    "text_sha": hashlib.sha256(r["text"].encode()).hexdigest()[:16], "text": r["text"],
                    "finish": r["finish"]})
    print(f"[{label}] single {i}: {singles[-1]['tok_s']:.2f} tok/s "
          f"({r['completion_tokens']} tok, finish={r['finish']})", flush=True)
res["single"] = singles

concs = []
for i in range(3):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(call, PROMPTS))
    wall = time.time() - t0
    total = sum(o["completion_tokens"] for o in outs)
    run = {
        "wall_s": wall,
        "total_tok": total,
        "agg_tok_s": total / wall,
        "per_req_latency_s": [o["elapsed_s"] for o in outs],
        "per_req_tok": [o["completion_tokens"] for o in outs],
        "finish": [o["finish"] for o in outs],
        "texts": [o["text"] for o in outs],
    }
    concs.append(run)
    print(f"[{label}] conc4 run {i}: agg {run['agg_tok_s']:.2f} tok/s, "
          f"wall {wall:.1f}s, tok {total}, finish {run['finish']}", flush=True)
res["conc4"] = concs

res["summary"] = {
    "single_median_tok_s": statistics.median(s["tok_s"] for s in singles),
    "conc4_median_agg_tok_s": statistics.median(c["agg_tok_s"] for c in concs),
    "conc4_median_wall_s": statistics.median(c["wall_s"] for c in concs),
}
json.dump(res, open(out_path, "w"), indent=2)
print(f"[{label}] SUMMARY {json.dumps(res['summary'])}", flush=True)
