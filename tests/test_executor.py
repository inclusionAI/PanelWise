from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from panelwise.errors import ExecutionError
from panelwise.executor import LocalShellExecutor


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = LocalShellExecutor(directory)
            result = await executor.execute("pwd")
            self.assertEqual(result.returncode, 0)
            self.assertEqual(Path(result.output.strip()), Path(directory).resolve())

    async def test_rejects_destructive_command_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = LocalShellExecutor(directory)
            with self.assertRaises(ExecutionError):
                await executor.execute("git reset --hard HEAD")
            with self.assertRaises(ExecutionError):
                await executor.execute("rm -fr generated")
            with self.assertRaises(ExecutionError):
                await executor.execute("   rm -fr generated")
            with self.assertRaises(ExecutionError):
                await executor.execute("env rm -fr generated")

    async def test_bounds_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = LocalShellExecutor(directory, max_output_chars=10)
            result = await executor.execute("printf 123456789012345")
            self.assertIn("truncated", result.output)

    async def test_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = LocalShellExecutor(directory, timeout_seconds=0.05)
            with self.assertRaisesRegex(ExecutionError, "timed out"):
                await executor.execute("sleep 1")

    async def test_finalize_includes_untracked_files_without_truncating_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "init", "--quiet", directory], check=True)
            new_file = Path(directory, "new file.txt")
            new_file.write_text("a long new-file payload\n", encoding="utf-8")

            executor = LocalShellExecutor(directory, max_output_chars=5)
            artifacts = await executor.finalize()

            self.assertIn("new file mode", artifacts["patch"])
            self.assertIn("a long new-file payload", artifacts["patch"])


if __name__ == "__main__":
    unittest.main()
