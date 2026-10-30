#!/usr/bin/env python3
"""Concurrent-burst greedy benchmark for the --serve batch path (4 slots).

Usage: batch_burst_bench.py CONFIG OUT_JSON [--rounds 2] [--max-tokens 256] [--conc 4]
Sends `conc` distinct prompts at once, `rounds` times, with temperature 0 and thinking off.
Records per request: completion tokens, wall seconds, sha256 of the reply text (no text is stored).
The engine must already be serving on the port in CONFIG's `port` field (default 8091).
The API key is read from CONFIG and never printed.
"""
import argparse, concurrent.futures, hashlib, json, time, urllib.request

PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a hash table handles collisions, with a small worked example.",
    "List five practical tips for keeping a home server secure, with one sentence each.",
    "Describe the water cycle to a ten year old in one paragraph.",
    "Write a haiku sequence of three poems about autumn rain on a city street.",
    "Summarize the causes of the French Revolution in six bullet points.",
    "Give a step by step recipe for a simple tomato soup for four people.",
    "Compare TCP and UDP in a short table-free paragraph for a new engineer.",
]

def one(cfg, prompt, max_tokens):
    body = {"model": cfg["model_name"], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + cfg["api_key"]}
    url = f"http://127.0.0.1:{cfg.get('port', 8091)}/v1/chat/completions"
    t0 = time.monotonic()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(),
                                                                 headers=headers), timeout=900).read())
    wall = time.monotonic() - t0
    text = d["choices"][0]["message"].get("content") or ""
    return {"tokens": d["usage"]["completion_tokens"], "wall_s": round(wall, 3),
            "finish": d["choices"][0]["finish_reason"],
            "sha256": hashlib.sha256(text.encode()).hexdigest()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--conc", type=int, default=4)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = {"rounds": []}
    for r in range(a.rounds):
        prompts = PROMPTS[(r * a.conc) % len(PROMPTS):][:a.conc]
        if len(prompts) < a.conc:
            prompts = (prompts + PROMPTS)[:a.conc]
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.conc) as ex:
            res = list(ex.map(lambda p: one(cfg, p, a.max_tokens), prompts))
        span = time.monotonic() - t0
        toks = sum(x["tokens"] for x in res)
        out["rounds"].append({"round": r, "span_s": round(span, 3), "agg_tok_s": round(toks / span, 2),
                              "requests": res})
        print(f"round {r}: {toks} tokens in {span:.2f} s = {toks / span:.1f} tok/s aggregate",
              [x["finish"] for x in res], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
