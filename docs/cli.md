# CLI reference

The `panelwise` executable is installed by the Python package.

## Initialize and validate

```bash
panelwise init
panelwise validate --config panelwise.yaml
panelwise --version
```

`init` never overwrites an existing file. `validate` parses the complete YAML and performs no model calls.

## Run a task

```bash
panelwise run "Compare two database designs" --eval
panelwise run "Fix the failing parser tests" --no-eval --workspace ./project
panelwise run --task-file task.md --eval --json
```

`--eval` and `--no-eval` override `execution.eval` in YAML for that invocation. `--workspace` similarly overrides `execution.workspace`.

| Option | Meaning |
|---|---|
| `--config PATH` | Configuration file; default `panelwise.yaml` |
| `--eval` | Independent complete attempts, evaluation, synthesis |
| `--no-eval` | Iterative proposals and actions in a shared environment |
| `--workspace PATH` | Fixed shell working directory for `--no-eval` |
| `--task-file PATH` | Read task text from a UTF-8 file |
| `--request-id ID` | Preserve a caller trace identifier in the result |
| `--json` | Emit the complete `PanelWiseResult` as JSON |

Task text may also be piped through stdin.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Completed result |
| `2` | Invalid arguments, configuration, credentials, or task input |
| `3` | Run returned `partial` or `failed` |
| `130` | Interrupted by the user |

