"""Regression tests for preserving assistant reasoning across multi-turn history."""
import tempfile
import unittest
from pathlib import Path

from serve.frontend import (ChatTemplate, _normalize_preserve_reasoning, openai_to_messages,
                             anthropic_to_messages)
from serve.responses import input_messages, template_kwargs
from serve.server import ByteTokenizer, MockEngine, Service, serve


ROOT = Path(__file__).resolve().parents[1]


class PreserveReasoning(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(ROOT / "serve/chat_template.jinja")

    def test_empty_reasoning_is_not_serialized_as_an_empty_thinking_turn(self):
        prompt = self.render(self.empty_history())
        self.assertNotIn("<think>\n\n</think>", prompt)
        self.assertIn("<|im_start|>assistant\nfirst answer", prompt)

    def empty_history(self):
        return [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer", "reasoning_content": ""},
            {"role": "user", "content": "second"},
        ]

    def render(self, history):
        return self.template.render(history, preserve_thinking=True, preserve_reasoning=True)


if __name__ == "__main__":
    unittest.main()
