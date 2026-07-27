from __future__ import annotations

import sys
import unittest
from unittest import mock

import fusion_full


class FusionDryRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_does_not_load_data_or_call_apis(self) -> None:
        with mock.patch.object(sys, "argv", ["fusion_full.py", "--dry-run"]), \
             mock.patch.object(fusion_full.config, "validate_provider_config") as validate, \
             mock.patch.object(fusion_full, "load_all", side_effect=AssertionError("data should not load")):
            await fusion_full.main()

        validate.assert_called_once_with(dry_run=True)


if __name__ == "__main__":
    unittest.main()
