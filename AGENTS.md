# Repository instructions

Read `CONTRIBUTING.md` before changing code.

## Product invariants

- `execution.eval=true`: independent full attempts, evaluator analysis, final synthesis.
- `execution.eval=false`: independent next-action proposals, one coordinated action, shared observable state.
- Task type is not hard-coded. Domain behavior belongs in prompts, clients, executors, or YAML.
- Public imports come from `panelwise.__all__`. Underscore-prefixed symbols are internal.
- Library code raises `PanelWiseError` subclasses; it must not call `sys.exit`.
- A failed panel member must not cancel successful members.
- The default local executor must reject obvious destructive commands and stay in its configured workspace.

## Validation

Run before handing off a change:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
.venv/bin/python -m build
.venv/bin/panelwise validate --config panelwise.example.yaml
```

Update the relevant README language and docs whenever public behavior changes. Do not add real API keys, benchmark task data, private trajectories, generated distributions, or local virtual environments.

Do not commit, push, publish, create releases, or alter remote state unless the user explicitly requests it.
