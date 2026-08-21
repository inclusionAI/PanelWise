# Python API reference

## Minimal integration

```python
import asyncio
from panelwise import PanelWise

async def main():
    async with PanelWise.from_yaml("panelwise.yaml", eval=True) as engine:
        result = await engine.run("Explain the trade-offs of event sourcing.")
    if result.ok:
        print(result.output)
    else:
        print(result.status, result.errors)

asyncio.run(main())
```

`PanelWise.from_yaml()` owns and closes its provider client. Advanced integrations can construct `PanelWise(config, client, executor=...)` with a custom client and retain lifecycle ownership.

## Stable public imports

The names exported from `panelwise.__all__` are the supported public API:

- `PanelWise`, `PanelWiseConfig`, `ProviderConfig`, `ModelConfig`, `ExecutionConfig`, `PromptConfig`, `load_config`, `config_from_dict`
- `ChatClient`, `OpenAICompatibleClient`
- `Executor`, `LocalShellExecutor`, `Observation`
- `PanelWiseResult`, `Candidate`, `StepRecord`, `Completion`, `Usage`, `RunMode`
- `PanelWiseError`, `ConfigurationError`, `ProviderError`, `ResponseError`, `ExecutionError`

Modules and names beginning with `_` are internal.

## `PanelWise.run`

```python
result = await engine.run(task, eval=None, request_id=None)
```

- `task` must be a non-empty string.
- `eval=None` uses YAML; `True` or `False` overrides it.
- `request_id` is a trace value returned unchanged. It is not an idempotency key.

Configuration and programmer errors raise documented `PanelWiseError` subclasses. Individual provider failures are isolated where possible and recorded in `result.errors`.

## Result contract

| Field | Both modes | Meaning |
|---|:---:|---|
| `request_id` | ✓ | Caller trace ID or generated UUID |
| `mode` | ✓ | `eval` or `trajectory` |
| `status` | ✓ | `completed`, `partial`, or `failed` |
| `output` | ✓ | Final answer or completion summary |
| `usage` | ✓ | Sum of provider-reported usage; zeros when absent |
| `errors` | ✓ | Non-fatal panel/stage failures and terminal run errors |
| `candidates` | eval | Independent full attempts |
| `evaluation` | eval | Parsed comparison object or raw evaluator response |
| `trajectory` | no-eval | Ordered proposals, decision, observation, and return code |
| `artifacts` | no-eval | Executor output such as `patch` and `workspace_status` |

`result.ok` is true only for `status == "completed"`. A completed result may still contain isolated panel or evaluator failures in `errors`; completion means PanelWise produced the mode's final output. `to_dict()` returns a JSON-ready representation.

## Evaluator schema

The default evaluator requests one JSON object with:

```json
{
  "consensus": [],
  "conflicts": [],
  "unique_insights": [],
  "blind_spots": [],
  "recommendation": ""
}
```

If JSON parsing fails, the raw response is preserved in `evaluation.raw`, the parse failure is recorded, and synthesis still proceeds.

## Trajectory action schema

Panel members and the coordinator use one of:

```json
{"kind":"command","command":"...","reason":"..."}
```

```json
{"kind":"done","answer":"...","reason":"..."}
```

The engine executes only the coordinator's command. A fenced Bash block is accepted as a compatibility fallback. A custom `Executor` can map the command string to a browser, database, simulator, remote worker, or another observable environment.

## Failure semantics

- Invalid YAML, missing credentials, empty tasks, and missing no-eval executors raise an exception before useful work starts.
- One panel member may fail without canceling the remaining panel.
- If every panel member fails, the result is `failed`.
- If evaluation fails, synthesis still receives the successful candidates.
- If synthesis fails, eval mode returns the first successful candidate with `partial` status.
- If trajectory mode reaches its limit but has a patch or answer, it returns `partial`; otherwise `failed`.
