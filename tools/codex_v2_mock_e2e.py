#!/usr/bin/env python3
# E2E: run the real codex CLI (multi_agent_v2) against the patched serve stack on a
# scripted MockEngine — no GPU. Scripts: parent spawns, subagent answers, parent closes.
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path.home() / "Strata-upstream"
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate             # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

SPAWN = ("Spawning now.\n</think>\n\n<tool_call>\n<function=collaboration.spawn_agent>\n"
         "<parameter=task_name>\narith\n</parameter>\n<parameter=message>\nWhat is 2+2? "
         "Reply with just the number.\n</parameter>\n</function>\n</tool_call>")
SUB = "Simple math.\n</think>\n\n4."
CLOSE = "The subagent computed it.\n</think>\n\nThe subagent's answer is 4."

tok = ByteTokenizer()
eng = MockEngine(tok, [SPAWN, SUB, CLOSE], max_context=262144)
httpd = serve(Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja")), port=8098)

repo = Path(tempfile.mkdtemp(prefix="amv2-e2e-", dir=str(ROOT.parent / ".hermes/cache/scratch")))
subprocess.run(["git", "init", "-q"], cwd=repo)
proc = subprocess.run(
    ["codex", "-c", "features.multi_agent_v2=true",
     "-c", "model_providers.strata.base_url=http://127.0.0.1:8098/v1",
     "exec", "--sandbox", "danger-full-access",
     "Use the collaboration tools: spawn a subagent named arith, ask it what 2+2 is, "
     "wait for its answer, and reply with that answer."],
    cwd=repo, capture_output=True, text=True, timeout=420)
httpd.shutdown()
out = proc.stdout + proc.stderr
fails = out.count("unsupported_parameter") + out.count("agent_message' are not supported")
print(f"codex exit={proc.returncode} agent_message-errors={fails}")
print("final codex text:", out.strip().splitlines()[-1] if out.strip() else "(empty)")
print("VERDICT:", "V2 ROUNDTRIP OK" if fails == 0 and proc.returncode == 0 else "V2 ROUNDTRIP FAILED")
sys.exit(0 if fails == 0 and proc.returncode == 0 else 1)
