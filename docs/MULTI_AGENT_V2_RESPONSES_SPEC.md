# Multi-agent v2: agent_message input items (Responses API)

## Purpose

codex CLI v0.166.0 with `features.multi_agent_v2=true` sends a new input item to
`/v1/responses`:

```json
{"type": "agent_message", "id": "...", "author": "...", "recipient": "...", "content": [...]}
```

`serve/responses.py` rejects it today. `input_messages()` walks the input items in a loop and
falls through to its final `else`, which raises `input items of type ... are not supported`. The
whole request fails, so a multi-agent run cannot start.

multi_agent v1 works because it never invents an item type. It routes every hop through the
namespaced `function_call` / `function_call_output` pair under the `collaboration.*` namespace,
which the Responses path already parses. v2 drops that route and speaks in a first-class item
instead, so the server has to learn the new shape.

This spec covers what the server must accept, how it must render it, and how we prove it works.

## The wire format

Two captures from a real v0.166.0 session. Keep them verbatim; the tests below replay them.

An outgoing task handed from the parent to a subagent:

```json
{"type": "agent_message", "id": "amsg_01a1a551-02c7-77b3-a326-066d0f5adebf", "author": "/root", "recipient": "/root/arith", "content": [{"type": "input_text", "text": "Message Type: NEW_TASK\nTask name: /root/arith\nSender: /root\nPayload:\n"}, {"type": "encrypted_content", "encrypted_content": "Question for you: what is 2+2? Reply with just the answer and nothing else."}]}
```

The reply back to the parent, carrying a failure:

```json
{"type": "agent_message", "id": "amsg_01a1a551-0e41-7490-9c27-69c5c0de83be", "author": "/root/arith", "recipient": "/root", "content": [{"type": "input_text", "text": "Message Type: FINAL_ANSWER\nTask name: /root\nSender: /root/arith\nPayload:\nAgent errored: {\"error\": {\"message\": \"input items of type 'agent_message' are not supported\", \"type\": \"invalid_request_error\", \"param\": \"input[6].type\", \"code\": \"unsupported_parameter\"}}\n\nThis agent's turn failed. If you still need this agent, use the available collaboration tools to give it another task."}]}
```

The fields, and what each one means:

- `id` is an opaque identifier with the `amsg_` prefix. The server does not parse it, echo it back,
  or deduplicate on it.
- `author` and `recipient` are agent paths in the task tree. `/root` is the top-level agent;
  `/root/arith` is a subagent spawned by it.
- `content` is a list of parts. Each part carries a `type`.

### What upstream says the parts can be

Upstream `codex-rs` `models.rs` defines:

```rust
ResponseItem::AgentMessage {
    id: Option<...>,
    author: String,
    recipient: String,
    content: Vec<AgentMessageInputContent>,
}
```

The official API reference lists `AgentMessage.content` as exactly three part types:
`input_text`, `input_image`, and `encrypted_content`. `encrypted_content` carries an opaque
string capped at 10,485,760 characters. (`codex-rs` shares its part enum with function-call
outputs, which also has `InputAudio`; the official agent-message schema does not list it. Do
not special-case audio; the tolerant fall-through below covers it.) The server accepts the
three official parts and stays tolerant of anything else in the list.

Optional item attributes, tolerated and ignored on input: `agent: {agent_name}` (the official
"produced by" marker) and `internal_chat_message_metadata_passthrough` (codex-rs). Read parts
by their `type`; ignore extra fields such as `prompt_cache_breakpoint` on `input_text`.

### The payload is not encrypted

For local providers the inter-agent payload arrives **plaintext** inside the `encrypted_content`
part. The field name is a leftover from the upstream encrypted-tool-content path; there is no
cipher on it here. This is capture-proven, not read off the code. Example 1 above is the evidence:
the `encrypted_content` string is a readable question in plain English.

Do not treat the field name as a promise about the value. Read what is there, and degrade if it is
not readable.

## Design

### Decision: add one branch to the `input_messages()` loop

Handle the new kind where the other kinds are handled, not in a pre-pass over the input list:

```python
elif kind == "agent_message":
    ...
```

A pre-pass would duplicate the turn bookkeeping the loop already does, and the copy would drift.
The loop is the single place that knows where a turn ends, so the new kind belongs there.

### Decision: close the open turn like any non-assistant message

Inside the branch, close `open_turn` and clear accumulated thinking exactly the way the existing
non-assistant branches do. An `agent_message` is a boundary between agent turns. If thinking stayed
open across it, the next assistant turn could inherit another agent's reasoning state.

### Decision: one chat message per item, never merged

Append exactly one chat message per `agent_message` item. Do not merge consecutive items, and do
not fold an item into a neighbouring message. Each hop in the task tree is a separate turn in the
conversation. Merging them loses the author and recipient boundaries that a subagent needs in order
to tell its own task from its sibling's.

### Decision: the role is `user`, through the existing rewriter

Use the existing `_late_system_to_user`-style handling so the message lands in place in the
transcript instead of at the end. Do not add a new role. A role the chat template has never seen is
a reliable way to break a prompt for every model at once.

### Decision: parts render through `_agent_message_parts()`

Add one helper that turns the part list into text. It maps each part type the way the existing
content renderers already do:

- `input_text` renders as-is. No rewrapping, no trimming.
- `input_image` renders like the image handling in `_content`.
- `encrypted_content`: if the value starts with our `ENCRYPTED_PREFIX`, decrypt it the way
  `_reasoning_text` does. Otherwise, if it decodes as text, render it as text. Otherwise emit the
  fixed placeholder line `[encrypted payload unavailable]`.

The helper must never raise. An undecodable part costs one placeholder line; it does not cost the
request. That is the whole point of this change, so the new failure mode has to be the opposite of
the one it replaces.

### Decision: the rendered layout

Render one `agent_message` as:

```text
[agent_message from AUTHOR to RECIPIENT]
<the input_text envelope>
<payload part 1>
<payload part 2>
```

The first line is fixed, and it gives the model the routing facts the item carries. The envelope is
the `input_text` part verbatim, which is where `Message Type`, `Task name`, and `Sender` live. The
payload parts follow, joined by newlines.

## Non-goals

The server never emits an `agent_message` output item. This change is input-only. codex sends the
item; it does not expect one back.

### Relation to the official hosted multi-agent (developers.openai.com, 2026-11)

The official Responses API has a *hosted* multi-agent mode: `"multi_agent": {"enabled": true}`
on the request, the SERVER executes six actions (`spawn_agent`, `send_message`, `followup_task`,
`wait_agent`, `interrupt_agent`, `list_agents`), and responses carry `multi_agent_call`,
`multi_agent_call_output` and `agent_message` as OUTPUT items, each with an `agent.agent_name`
attribute. This spec implements NONE of that. codex's `multi_agent_v2` against a custom provider
orchestrates on the CLIENT: collaboration actions run as ordinary namespaced `function_call` /
`function_call_output` under `collaboration.*` (already supported), and only the inter-agent
`agent_message` INPUT item (identical shape to the official one, parts included) reaches us. If
hosted mode is ever wanted, that is a separate feature with its own spec; do not half-build it
here.

Out of scope, and deliberately untouched:

- `check_request` validation.
- The streaming path.
- The chat path.
- `request_tools`. The `collaboration.*` namespace tools already parse, and v1 depends on them.

## Tests

Add these to `serve/test_responses.py`. Use `unittest`, and mirror the multi_agent v1 tests at
lines 56-69 and 751 so the two versions read the same way side by side.

`test_agent_message_item_parses`
: Feed in example 1 and assert the `messages` shape it produces.

`test_agent_message_encrypted_ciphertext_degrades`
: Feed in an undecryptable `encrypted_content` value, then assert the placeholder line and that no
  exception is raised.

`test_agent_message_prompt_golden`
: Assert that `[agent_message from /root to /root/arith]` appears in the rendered prompt.

`test_agent_message_between_turns_keeps_tool_pairing`
: Put an `agent_message` between a `function_call` and its `function_call_output`, and assert that
  pairing through `_order_tool_results` is unaffected. This is the regression most likely to bite:
  the new branch closes a turn, and a turn closed in the wrong order would separate a tool call
  from its output.

## Acceptance

1. The `serve` tests pass under both runners:

   ```sh
   python3 -m pytest serve/test_responses.py
   python3 -m unittest serve.test_responses
   ```

2. A real client completes a roundtrip against the dev endpoint:

   ```sh
   codex -c features.multi_agent_v2=true
   ```

   The run must take `spawn_agent` -> subagent reply -> `FINAL_ANSWER` with no
   `unsupported_parameter` errors.

3. The multi_agent v1 tests stay green. v1 is the supported path today; nothing here may disturb it.

## Cross-reference

Captures, the reproduction steps that produced them, and the client version they came from are in
[docs/AGENT_MESSAGE_EVIDENCE.md](AGENT_MESSAGE_EVIDENCE.md). Every claim in this spec about the
wire format traces back to that file. Claims that come from reading upstream code, and not from a
capture, are marked as such above.
