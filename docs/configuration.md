# Configuration

PanelWise uses one versioned YAML file for both execution modes. CLI flags override YAML without editing it.

```yaml
version: 1

provider:
  name: openrouter
  api_key_env: OPENROUTER_API_KEY
  timeout_seconds: 120
  retries: 2
  headers: {}
  request_options: {}
  eval_request_options: {}
  trajectory_request_options: {}

models:
  panel: [z-ai/glm-5.1, minimax/minimax-m3, qwen/qwen3.7-max]
  coordinator: z-ai/glm-5.1
  evaluator: z-ai/glm-5.1

execution:
  eval: true
  concurrency: 3
  max_steps: 24
  workspace: .
  command_timeout_seconds: 120
  max_output_chars: 8000
  allow_unsafe_commands: false
```

The complete commented template is [`panelwise.example.yaml`](../panelwise.example.yaml).

## Mode selection

- `execution.eval: true` runs independent full attempts, an evaluator, and a synthesizer. This topology fits answer-centric tasks, including deep research.
- `execution.eval: false` repeatedly gathers next-action proposals, chooses one action, executes it against shared state, and feeds the observation into the next round. This topology fits stateful tasks, including coding.

The topology does not restrict the task type. Custom prompts, clients, and executors can adapt either mode to another domain.

## Providers

| `provider.name` | Default base URL | Default key variable |
|---|---|---|
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `zenmux` | `https://zenmux.ai/api/v1` | `ZENMUX_API_KEY` |
| `openai-compatible` | Required in YAML | `PANELWISE_API_KEY` |

`headers` adds non-secret provider headers. `request_options` is merged into every chat-completion payload. `eval_request_options` and `trajectory_request_options` are applied only in their corresponding modes and override common keys. Do not commit API keys to YAML.

The engine owns `model`, `messages`, and `stream`, so request options cannot override them. Likewise, `headers` cannot replace `Authorization`, `Content-Type`, or `User-Agent`; use a custom `ChatClient` when a gateway requires a different authentication protocol.

For example, OpenRouter eval mode can reproduce a tool-enabled deep-research topology with server-managed search and fetch:

```yaml
provider:
  name: openrouter
  eval_request_options:
    tools:
      - type: openrouter:web_search
      - type: openrouter:web_fetch
```

These tool identifiers are OpenRouter-specific and should not be sent to ZenMux or another compatible gateway unless it documents the same extension.

For a provider whose protocol is not OpenAI-compatible, implement the documented `ChatClient` protocol and pass it to `PanelWise`.

## Model roles

- `panel`: two or more models that independently answer or propose actions.
- `coordinator`: synthesizes evaluated answers or chooses the next shared action.
- `evaluator`: compares full answers in eval mode. When omitted, it defaults to the coordinator.

Model identifiers are forwarded unchanged to the provider. Panel models may be identical if repeated sampling is desired.

## Prompts

The optional `prompts` mapping can replace any of five built-in prompts: `eval_panel`, `evaluator`, `synthesizer`, `trajectory_panel`, and `coordinator`. Replacements are complete prompt substitutions, not appended fragments. Preserve the evaluator JSON keys and trajectory action schema documented in [API reference](./api.md) unless a custom engine layer parses another format.

## Safety boundary

Trajectory mode starts model-selected shell commands from `execution.workspace`. The default executor blocks a short list of obviously destructive commands and bounds runtime and output, but it does not provide filesystem or network isolation. This is a guardrail, not a sandbox. Use a disposable container or VM for untrusted tasks. `allow_unsafe_commands: true` should only be used inside such an isolated environment.
