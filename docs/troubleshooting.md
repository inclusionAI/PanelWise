# Troubleshooting

## `panelwise: missing provider API key`

Set the environment variable named by `provider.api_key_env`:

```bash
export OPENROUTER_API_KEY="..."
```

PanelWise does not automatically load `.env`. Use your shell, process manager, or a dotenv library in the host application.

## HTTP 401 or 403

Verify the key, provider base URL, account permissions, and model access. A provider's model slug is not portable unless that provider documents it.

## HTTP 429 or transient 5xx failures

PanelWise retries 408, 409, 425, 429, and 5xx responses according to `provider.retries`. Reduce `execution.concurrency`, raise provider limits, or add application-level backoff for repeated runs.

## Provider returns no choices or empty content

The built-in client raises `ResponseError`. Inspect the provider's raw logs and remove incompatible `request_options`. ZenMux and OpenRouter may support different provider-specific fields even though both expose OpenAI-compatible chat completions.

## Evaluator output is not JSON

The run continues. `result.evaluation.raw` preserves the text and `result.errors` records the parse failure. Use a model with reliable JSON output or specialize `prompts.evaluator` while preserving the documented keys.

## `--no-eval requires an executor or workspace`

CLI users should pass `--workspace`. Python users can call `PanelWise.from_yaml(..., eval=False, workspace="...")` or inject a custom `Executor`.

## A command was rejected

The default executor rejects obvious destructive commands. Do not disable this protection on a normal workstation. If the task genuinely requires such operations, use a disposable container or VM and set `allow_unsafe_commands: true` inside that isolated environment.

## Trajectory mode stops with `partial`

Inspect `result.trajectory`, `result.errors`, and `result.artifacts`. Raise `execution.max_steps` only after checking whether models are repeating reads, producing invalid JSON actions, or receiving truncated command output.

## Verify the installation

```bash
panelwise --version
panelwise validate --config panelwise.yaml
python -c "import panelwise; print(panelwise.__version__)"
```

For source checkouts, reinstall after dependency or packaging changes:

```bash
python -m pip install -e '.[dev]'
```

