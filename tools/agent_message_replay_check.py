#!/usr/bin/env python3
# Independent verifier: replay the REAL captured multi_agent_v2 requests (from the
# amo-capture JSONL) through the patched serve stack on a MockEngine HTTP endpoint.
# Proves: request that got 400 pre-change now returns 200, and the rendered prompt
# carries the agent_message lines. No GPU, no pack.
import json, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path.home() / "Strata-upstream"
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate            # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

CAP = Path.home() / ".hermes/cache/scratch/amo-capture/requests.jsonl"
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")

with open(CAP, "rb") as f:
    recs = [json.loads(line) for line in f]
with_am = [r for r in recs if any(isinstance(i, dict) and i.get("type") == "agent_message"
                                  for i in r["body"].get("input", []))]
print(f"capture: {len(recs)} requests, {len(with_am)} contain agent_message items")
assert with_am, "no agent_message captures found"

tok = ByteTokenizer()
eng = MockEngine(tok, "done\n</think>\n\nok.", max_context=262144)
httpd = serve(Service(eng, tok, TEMPLATE), port=8097)
fails = 0
for n, rec in enumerate(with_am):
    body = dict(rec["body"])
    body["stream"] = False
    body.pop("include", None)                    # mock may not know include semantics
    req = urllib.request.Request("http://127.0.0.1:8097/v1/responses",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        status = out.get("status")
        types = [i.get("type") for i in out.get("output", [])]
    except urllib.error.HTTPError as e:
        status, types = f"HTTP {e.code}", e.read()[:200].decode()
    prompt = tok.decode(eng.last_prompt) if eng.last_prompt else ""
    marker = "[agent_message from /root to /root/arith]" in prompt or \
             "[agent_message from /root/arith to /root]" in prompt
    ok = status == "completed" and marker
    print(f"req {n}: status={status} output={types if isinstance(types, list) else types!r}[:80] "
          f"agent_message-in-prompt={marker} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        fails += 1
httpd.shutdown()
print("VERDICT:", "ALL CAPTURED v2 REQUESTS SERVE" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
