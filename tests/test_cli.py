from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from panelwise.cli import main


class CliTests(unittest.TestCase):
    def test_init_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panelwise.yaml"
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["init", str(path)]), 0)
            self.assertTrue(path.exists())
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["validate", "--config", str(path)]), 0)
            self.assertIn("Valid configuration", output.getvalue())

    def test_init_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panelwise.yaml"
            path.write_text("existing", encoding="utf-8")
            errors = StringIO()
            with redirect_stderr(errors):
                self.assertEqual(main(["init", str(path)]), 2)
            self.assertIn("refusing to overwrite", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
