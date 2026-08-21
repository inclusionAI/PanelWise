from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from panelwise.config import config_from_dict, load_config
from panelwise.errors import ConfigurationError


class ConfigTests(unittest.TestCase):
    def test_loads_two_mode_configuration(self) -> None:
        config = config_from_dict(
            {
                "version": 1,
                "provider": {
                    "name": "zenmux",
                    "eval_request_options": {"temperature": 0.2},
                },
                "models": {"panel": ["a", "b"], "coordinator": "c"},
                "execution": {"eval": False, "max_steps": 5},
            }
        )
        self.assertEqual(config.provider.base_url, "https://zenmux.ai/api/v1")
        self.assertEqual(config.provider.api_key_env, "ZENMUX_API_KEY")
        self.assertEqual(config.provider.eval_request_options["temperature"], 0.2)
        self.assertFalse(config.execution.eval)
        self.assertEqual(config.models.evaluator, "c")
        self.assertTrue(config.with_eval(True).execution.eval)

    def test_reads_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panelwise.yaml"
            path.write_text(
                "version: 1\nmodels:\n  panel: [a, b]\n  coordinator: c\n",
                encoding="utf-8",
            )
            self.assertEqual(load_config(path).models.panel, ("a", "b"))

    def test_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "models.panel"):
            config_from_dict({"version": 1, "models": {"coordinator": "c"}})
        with self.assertRaisesRegex(ConfigurationError, "execution.eval"):
            config_from_dict(
                {
                    "version": 1,
                    "models": {"panel": ["a", "b"], "coordinator": "c"},
                    "execution": {"eval": "yes"},
                }
            )
        with self.assertRaisesRegex(ConfigurationError, "execution.workspace"):
            config_from_dict(
                {
                    "version": 1,
                    "models": {"panel": ["a", "b"], "coordinator": "c"},
                    "execution": {"workspace": None},
                }
            )
        with self.assertRaisesRegex(ConfigurationError, "cannot override"):
            config_from_dict(
                {
                    "version": 1,
                    "provider": {"request_options": {"messages": []}},
                    "models": {"panel": ["a", "b"], "coordinator": "c"},
                }
            )


if __name__ == "__main__":
    unittest.main()
