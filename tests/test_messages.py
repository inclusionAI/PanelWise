from __future__ import annotations

import copy
import unittest

from draco_eval.messages import normalize_messages


class NormalizeMessagesTests(unittest.TestCase):
    def test_string_becomes_one_user_message(self) -> None:
        self.assertEqual(normalize_messages("hello"), [{"role": "user", "content": "hello"}])

    def test_list_is_deep_copied_without_rewriting_provider_fields(self) -> None:
        source = [{
            "role": "developer",
            "content": [{"type": "input_text", "text": "hello"}],
            "provider_field": {"nested": ["keep"]},
        }]
        normalized = normalize_messages(source)
        self.assertEqual(normalized, source)
        self.assertIsNot(normalized, source)
        self.assertIsNot(normalized[0], source[0])
        normalized[0]["provider_field"]["nested"].append("changed")
        self.assertEqual(source[0]["provider_field"]["nested"], ["keep"])

    def test_empty_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "messages must not be empty"):
            normalize_messages([])

    def test_unsupported_top_level_type_is_rejected(self) -> None:
        for value in (None, {"role": "user"}, 3):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "prompt string or messages list"):
                    normalize_messages(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
