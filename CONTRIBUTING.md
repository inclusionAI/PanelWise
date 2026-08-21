# Contributing to PanelWise

PanelWise keeps a small core: one provider-neutral aggregation engine, two execution topologies, and explicit extension protocols. Contributions should strengthen that core or live behind a client, executor, configuration, or prompt extension point.

## Before opening a pull request

Open an issue first for:

- public API or result-contract changes;
- a new execution mode or core abstraction;
- provider-specific behavior in the generic client;
- changes to command execution or safety policy;
- large refactors or new dependencies.

Small documentation corrections and focused tests may go directly to a pull request. If uncertain, open a short issue describing the problem, why it matters, and the intended scope.

## Development setup

```bash
git clone https://github.com/inclusionAI/PanelWise.git
cd PanelWise
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m build
```

No model key is required for the test suite.

## Pull request workflow

1. Fork the repository and create a focused branch from current `main`.
2. Add or update offline tests for every behavior change.
3. Update README or `docs/` when the CLI, YAML, public API, errors, or result fields change.
4. Run the complete tests and package build.
5. Push the branch and open a pull request using the repository template.

Use a conventional, concise PR title such as:

- `feat: add a custom executor hook`
- `fix: preserve partial panel failures`
- `docs: clarify ZenMux configuration`
- `test: cover malformed coordinator actions`

Keep one problem per PR. Explain what changed, why it fixes the issue, and exactly how it was verified. Large generated narratives make review harder; concise technical evidence is preferred.

## Required evidence

For deterministic changes, include the test name and command. For model-dependent behavior, include a minimal redacted reproduction with:

- PanelWise version and commit;
- mode (`--eval` or `--no-eval`);
- model and provider identifiers;
- relevant YAML fields and CLI overrides;
- request ID;
- final status and the shortest trajectory excerpt that demonstrates the behavior.

Never post API keys, authorization headers, private prompts, customer data, or an unredacted workspace trajectory.

## Provider changes

Prefer the existing OpenAI-compatible client and YAML options. A new client implementation must test:

- authentication and base-URL behavior;
- successful text and usage parsing;
- non-retryable 4xx errors;
- 429/5xx retry behavior;
- malformed and empty successful responses;
- cleanup of async resources.

Do not add a provider-specific branch to the engine when a `ChatClient` adapter can contain it.

## Executor changes

Executor changes require tests for working-directory initialization, timeout behavior, output bounds, failure reporting, and destructive-command handling. The default must retain conservative guardrails for a normal developer workstation; broader command access belongs in a disposable environment.

## AI-assisted contributions

AI assistance is allowed, but the contributor must understand and be able to explain every change. Review generated code, run the tests yourself, remove irrelevant edits, and do not submit generated issue or PR text without checking its accuracy.

## Release changes

Only maintainers publish releases. Do not change package versions or release workflows in an unrelated PR. See [docs/releasing.md](./docs/releasing.md).
