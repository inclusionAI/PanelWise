"""Stable command-line interface for PanelWise."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from importlib import resources
from pathlib import Path

from . import __version__
from .client import client_from_config
from .config import load_config
from .engine import PanelWise
from .errors import PanelWiseError
from .executor import LocalShellExecutor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panelwise",
        description=(
            "Fuse multiple model perspectives into an evaluated answer or shared trajectory."
        ),
    )
    parser.add_argument("--version", action="version", version=f"panelwise {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one arbitrary task")
    run.add_argument("task", nargs="?", help="task text; omit when using --task-file")
    run.add_argument("--task-file", type=Path, help="read task text from a UTF-8 file")
    run.add_argument("--config", default="panelwise.yaml", help="YAML configuration path")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        help="independent answers → evaluation → synthesis",
    )
    mode.add_argument(
        "--no-eval",
        dest="eval",
        action="store_false",
        help="shared proposal → decision → execution trajectory",
    )
    run.set_defaults(eval=None)
    run.add_argument("--workspace", help="shared workspace for --no-eval")
    run.add_argument("--request-id", help="caller-provided trace identifier")
    run.add_argument("--json", action="store_true", help="write the complete result as JSON")

    validate = sub.add_parser("validate", help="validate configuration without model calls")
    validate.add_argument("--config", default="panelwise.yaml")

    init = sub.add_parser("init", help="write a documented starter YAML")
    init.add_argument("path", nargs="?", default="panelwise.yaml")
    return parser


def _task(args: argparse.Namespace) -> str:
    if args.task and args.task_file:
        raise PanelWiseError("provide task text or --task-file, not both")
    if args.task_file:
        try:
            return args.task_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise PanelWiseError(f"cannot read task file: {exc}") from exc
    if args.task:
        return args.task
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise PanelWiseError("task text is required")


async def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.eval is not None:
        config = config.with_eval(args.eval)
    if args.workspace:
        config = config.with_workspace(args.workspace)

    executor = None
    if not config.execution.eval:
        executor = LocalShellExecutor(
            config.execution.workspace,
            timeout_seconds=config.execution.command_timeout_seconds,
            max_output_chars=config.execution.max_output_chars,
            allow_unsafe_commands=config.execution.allow_unsafe_commands,
        )
    async with client_from_config(config.provider) as client:
        result = await PanelWise(config, client, executor=executor).run(
            _task(args), eval=config.execution.eval, request_id=args.request_id
        )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.output)
        if result.errors:
            print("\nWarnings:", file=sys.stderr)
            for error in result.errors:
                print(f"- {error}", file=sys.stderr)
    return 0 if result.status == "completed" else 3


def _init(path_value: str) -> int:
    path = Path(path_value)
    if path.exists():
        raise PanelWiseError(f"refusing to overwrite existing file: {path}")
    template = (
        resources.files("panelwise").joinpath("panelwise.example.yaml").read_text(encoding="utf-8")
    )
    path.write_text(template, encoding="utf-8")
    print(f"Wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            config = load_config(args.config)
            mode = "eval" if config.execution.eval else "trajectory"
            print(f"Valid configuration: provider={config.provider.name}, mode={mode}")
            return 0
        if args.command == "init":
            return _init(args.path)
        return asyncio.run(_run(args))
    except PanelWiseError as exc:
        print(f"panelwise: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("panelwise: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
