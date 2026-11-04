# Evidence pack: wiring `agent_message` into the Responses API (multi_agent_v2)

Collected 2026-11-04 on cachy-serv. Sources: live captures through a transparent logging proxy
(codex 0.166.0, `features.multi_agent_v2=true` -> :8099 -> Strata serve :8086), and upstream
openai/codex `codex-rs/protocol/src/models.rs` + `protocol.rs` via Context7. Everything below is
VERIFIED by capture unless marked (upstream struct) or (unverified).

## 1. The failure

With `features.multi_agent_v2=true`, every request codex sends to Strata after the first
inter-agent message is rejected at input validation:

```
{"error":{"message":"input items of type 'agent_message' are not supported",
 "type":"invalid_request_error","param":"input[11].type","code":"unsupported_parameter"}}
```

Raised in `serve/responses.py` `input_messages()` (the final `else:` of the input-item loop,
line ~200 on branch `feat/responses-agent-message`). v1 mode (`multi_agent_v2=false`) works
today because its spawn tool returns results through ordinary `function_call_output` items.

## 2. Exact wire shape (from capture, codex 0.166.0)

Parent request input items seen: `message` x25, `reasoning` x3, `function_call` x2,
`function_call_output` x2, `additional_tools` x1, `agent_message` x2. Top-level request params:
`model, input, instructions? (absent), tools, tool_choice, parallel_tool_calls, reasoning,
stream, store, include, text, prompt_cache_key, client_metadata`.

The two captured `agent_message` items, verbatim:

```json
{"type": "agent_message",
 "id": "amsg_01a1a551-02c7-77b3-a326-066d0f5adebf",
 "author": "/root",
 "recipient": "/root/arith",
 "content": [
  {"type": "input_text",
   "text": "Message Type: NEW_TASK\nTask name: /root/arith\nSender: /root\nPayload:\n"},
  {"type": "encrypted_content",
   "encrypted_content": "Question for you: what is 2+2? Reply with just the answer and nothing else."}
 ]}
```

```json
{"type": "agent_message",
 "id": "amsg_01a1a551-0e41-7490-9c27-69c5c0de83be",
 "author": "/root/arith",
 "recipient": "/root",
 "content": [
  {"type": "input_text",
   "text": "Message Type: FINAL_ANSWER\nTask name: /root\nSender: /root/arith\nPayload:\nAgent errored: {\"error\": {\"message\": \"input items of type 'agent_message' are not supported\", \"type\": \"invalid_request_error\", \"param\": \"input[6].type\", \"code\": \"unsupported_parameter\"}}\n\nThis agent's turn failed. If you still need this agent, use the available collaboration tools to give it another task."}
 ]}
```

Notes from capture:
- `author`/`recipient` are agent paths (`/root` = main agent, `/root/<task_name>` = subagent).
- Content parts are the same part objects as function-call outputs: `input_text`, and
  `encrypted_content` whose value is a plain string. In this local capture the
  "encrypted_content" is NOT encrypted — codex put the plaintext task payload in it.
- The `input_text` part is a human-readable envelope ("Message Type: NEW_TASK|FINAL_ANSWER\n
  Task name: ...\nSender: ...\nPayload:\n"); the payload follows as the next part.
- The failing spawn call was `function_call` namespace `collaboration`, name `spawn_agent`,
  with args including `task_name`, `message`, `fork_turns`, `model`, `reasoning` — those parts
  already parse fine today (namespaced function_call is supported, see tests at
  `serve/test_responses.py:56-69,602-618` using namespace `multi_agent_v1`).
- `id` prefix is `amsg_`.

## 3. Upstream structs (openai/codex codex-rs, via Context7)

```rust
ResponseItem::AgentMessage {
    id: Option<ResponseItemId>,
    author: String,        // agent path
    recipient: String,
    content: Vec<AgentMessageInputContent>,
    internal_chat_message_metadata_passthrough: Option<...>,
}  // serde tag "type", snake_case -> wire type name "agent_message"
```

`AgentMessageInputContent` tracks `FunctionCallOutputContentItem`:
`InputText{text}`, `InputImage{...}`, `InputAudio{audio_url}`, `EncryptedContent{encrypted_content}`.

`InterAgentCommunication { author, recipient, other_recipients: Vec<AgentPath>, content: String,
encrypted_content: Option<String>, trigger_turn: bool }` is the internal protocol message;
`to_model_input_item()` produces the `agent_message` input item.
(upstream struct) `to_response_input_item()` can instead produce a plain `message` with role
assistant — i.e. codex itself sometimes renders inter-agent text as an ordinary message.
Whether codex uses that fallback for non-OpenAI providers: (unverified).

Also (upstream, `core/src/client.rs`): for non-OpenAI providers codex *clears*
`encrypted_content` on most items but the agent_message content part above was captured with
it intact — clearing evidently does not strip `content[]` parts of agent_message. (unverified
whether it is always plaintext for local providers, or ciphertext when the OpenAI backend
encrypted it upstream; design must handle both.)

## 4. Where Strata changes

`serve/responses.py`:
- `input_messages()` input-item loop (~154-201): add an `agent_message` kind before the reject.
- `_content()` (~57-85) rejects parts of type `encrypted_content` — either extend it or give
  agent_message its own part renderer.
- Role/turn semantics: the loop keeps `open_turn` (assistant turn that later function_calls
  join) and `thinking` (reasoning awaiting its assistant turn). A new kind must decide how it
  interacts with both. Tool results are reordered by `_order_tool_results()` (213-233);
  `_late_system_to_user()` (~end) turns late system messages into user messages.
- `check_request()` needs no change (top-level params are tolerated).
- No output-side change: Strata never emits `agent_message` response items; codex synthesizes
  them from the collaboration tool results (function_call_output) it already understands.
- `request_tools()` already accepts `type:"namespace"` tool groups (verified by the
  multi_agent_v1 tests), so the `collaboration.*` tool family needs no schema work.

Test surface: `serve/test_responses.py` (unittest; golden-prompt assertions like
`:751` `assertIn("multi_agent_v1.spawn_agent", prompt...)`).

## 5. Open design decisions for the spec

1. Which chat-template role carries an agent_message, and how the envelope+payload render
   (join parts? keep "Message Type/Sender" header? preserve part boundaries?).
2. Whether the item closes the current `open_turn` / flushes `thinking` like `message` does.
3. What to do when `encrypted_content` is real ciphertext (undecryptable by Strata): drop it
   with the plaintext envelope kept, or reject the item — must not crash the prompt build.
4. Whether back-to-back agent_messages merge; where they sit relative to the
   leading-system-message collapse (`input_messages` end) and `_late_system_to_user`.
5. Prompt-injection stance: agent_message text is model-authored content from another thread;
   the main agent's developer instructions in the captured prompt already say untrusted data
   must not be followed. Strata treats it as ordinary user-visible text (no special escaping) —
   spec should say so explicitly.

## 6. Reproduction (exact)

```bash
# proxy at tools/agent_message_capture_proxy.py forwards :8099 -> :8086 and logs bodies
python3 tools/agent_message_capture_proxy.py &
cd "$(mktemp -d)" && git init -q
codex -c features.multi_agent_v2=true \
      -c model_providers.strata.base_url="http://localhost:8099/v1" \
      exec --sandbox danger-full-access \
      "Spawn one subagent. Ask it (via the multi-agent message tools) what 2+2 is, wait for its reply, and report the answer."
# captured JSONL: ~/.hermes/cache/scratch/amo-capture/requests.jsonl
```
