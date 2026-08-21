<p align="center">
  <img src="./assets/panelwise-logo.svg" alt="PanelWise logo" width="720">
</p>

<p align="center">
  <strong>English</strong> · <a href="./README.zh-CN.md">中文</a>
</p>

<p align="center">
  <a href="https://github.com/inclusionAI/PanelWise/actions/workflows/ci.yml"><img src="https://github.com/inclusionAI/PanelWise/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache 2.0 license"></a>
</p>

# PanelWise

**Fuse the complementary strengths of multiple models into one stronger, verifiable result.**

PanelWise starts with a simple question: if every weaker model produces an answer with something worth keeping, can we combine the complementary parts of those answers into a result that outperforms a stronger individual model? PanelWise lets multiple models examine the same problem independently and propose different analyses or next actions. It then organizes those judgments into a shared, executable workflow and submits the final result to an independent grader for verification.

<p align="center">
  <img src="./assets/panelwise-flow-concept.png" alt="Three different model perspectives flow through PanelWise into one fused answer" width="1000">
</p>

PanelWise applies this idea to two workflows:

- **Deep research:** independent agents research the same question, a judge identifies consensus and blind spots, and a synthesizer writes one evidence-grounded report.
- **Software engineering:** multiple agents propose the next action against the same live repository, PanelWise executes one merged action, and every subsequent round sees the resulting state.

## How it works

### Deep research

```text
One research question
        ↓
Independent research agents × N
        ↓
Consensus · contradictions · unique evidence · blind spots
        ↓
One synthesized report
        ↓
Independent rubric grader
```

Each research agent runs its own multi-step ReAct loop with search and page retrieval. PanelWise can preserve the research trajectories; it compares the reports structurally and grounds the final answer in the evidence the panel collected.

### Software engineering

```text
Task specification + real repository
        ↓
Multiple models propose the next action
        ↓
PanelWise merges and executes one action
        ↓
Shared repository state for the next round
        ↓
Git patch → independent test harness
```

The coding workflow operates on a real working tree instead of asking models to imagine an entire diff. Models can therefore react to actual source files, command output, earlier edits, and test results. The final patch comes from `git diff` and can be evaluated by the benchmark's native harness.

## Results on DRACO

We evaluated PanelWise on all 100 tasks in [DRACO](https://arxiv.org/abs/2503.14476), a deep-research benchmark spanning ten domains. Each task is scored against a weighted rubric covering factual accuracy, breadth, depth, presentation, and citation quality.

<p align="center">
  <img src="./assets/panelwise-draco-results.png" alt="PanelWise DRACO benchmark results compared with published OpenRouter Fusion and Claude Fable 5 results" width="1000">
</p>

The **OpenRouter Fusion score of 68.3** and the **Claude Fable 5 score of 65.3** shown in the chart above and table below are both quoted from [OpenRouter's official Fusion launch post](https://openrouter.ai/blog/announcements/fusion-beats-frontier/). PanelWise scores come from our complete 100-task benchmark runs.

| System | Models | DRACO score |
|---|---|---:|
| **PanelWise frontier panel** | Opus 4.8 + GPT-5.5 + Gemini 3.1 Pro | **73.68** |
| OpenRouter Fusion, published comparison | Opus 4.8 + GPT-5.5 + Gemini 3.1 Pro | 68.3 |
| **PanelWise budget panel** | GLM 5.1 + MiniMax M3 + Qwen 3.7 Max | **66.42** |
| Strongest published single-model baseline | Claude Fable 5 | 65.3 |

The frontier run scored **5.38 points above** OpenRouter's published result for the same three-model panel. The fully independent budget run scored **1.12 points above** the strongest published single-model baseline.

OpenRouter's Fable 5 score reflects the 93 tasks it completed because content filters blocked seven tasks.

A panel-swap ablation replaced Qwen with Gemini 3.5 Flash and reached **70.95**. That experiment reused existing GLM and MiniMax reports, so we report it as evidence that model diversity can matter more than standalone ranking rather than as a fully independent end-to-end run.

External comparison source: [OpenRouter's official Fusion launch post](https://openrouter.ai/blog/announcements/fusion-beats-frontier/).

## A real coding case: combining the insights of smaller models

We also tested the same idea on the SWE-Bench Pro task **“Proper WebFinger Response for Instance Actor”**:

```text
instance_NodeBB__NodeBB-da0211b1a001d45d73b4c84c6417a4f1b0312575-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
```

NodeBB could resolve individual users through WebFinger, but it could not correctly resolve the site itself as an ActivityPub Application actor. A valid fix had to coordinate two controllers while preserving existing user behavior:

1. Use the bare `hostname` as the Application actor's `preferredUsername`, while using the configured site title as its display name.
2. Keep `host`—including an optional port—for WebFinger address validation, but use `hostname` to recognize the instance actor.
3. Resolve the instance before ordinary-user permissions and UID lookup.
4. Preserve canonical user slugs, profile links, UID actor links, permission checks, and 404 behavior for normal users.

GLM 5.1, MiniMax M3, and Qwen 3.7 Max each found part of the solution. Their blind spots were different: one conflated `host` with `hostname`, another left instance discovery behind user permissions, and another changed the instance path while leaving a non-canonical ordinary-user link. PanelWise carried those partial judgments through a shared execution trajectory and converged on the two-controller fix.

The important behavior is the sequence of corrections: locate both protocol entry points, separate address validation from actor identity, move the instance path ahead of user logic, revisit compatibility for existing users, and finally align the Application actor representation.

## Quick start

### Requirements

- Python 3.11 recommended; the core package also runs on Python 3.9
- An [OpenRouter](https://openrouter.ai/) or ZenMux API key
- An [Exa](https://exa.ai/) API key when using the default direct-search backend
- Docker and the SWE-Bench Pro harness for coding evaluation

### Install

```bash
git clone https://github.com/inclusionAI/PanelWise.git
cd PanelWise

python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` and `EXA_API_KEY` in `.env`. To use ZenMux instead, set `MODEL_PROVIDER=zenmux`, add `ZENMUX_API_KEY`, and keep `RESEARCH_SEARCH_BACKEND=exa`.

### Validate the setup without external calls

```bash
.venv/bin/python fusion_full.py --dry-run
```

The dry run checks local configuration without loading benchmark data, calling a model or search service, invoking a grader, or creating output.

### Run one deep-research task

```bash
.venv/bin/python fusion_full.py --limit 1
```

Results are written incrementally under `output/`, so interrupted runs can resume.

### Run the complete deep-research evaluation

```bash
.venv/bin/python fusion_full.py
```

The root-level ablation and comparison scripts are preserved for research transparency. They may depend on historical model slugs, optional services, or intermediate artifacts and are not stable public interfaces.

## Evaluation and reproducibility notes

- External numbers are cited as published comparison points. Differences in model snapshots, search configuration, grader versions, and retry policy can affect absolute comparability.
- `requirements.txt` contains core runtime dependencies. The optional official grader is pinned separately in `requirements-eval.txt` and requires Python 3.10 or newer.
- Direct local page retrieval is enabled by default for trusted local research runs. Set `RESEARCH_ENABLE_DIRECT_FETCH=0` to disable it. This is an opt-out, not an SSRF sandbox.

## Contributors

<p>
  <a href="https://github.com/jcguo123"><img src="https://avatars.githubusercontent.com/u/164945525?v=4" width="72" alt="Jiacheng Guo (@jcguo123)" title="Jiacheng Guo (@jcguo123)"></a>
  <a href="https://github.com/DPLL"><img src="https://avatars.githubusercontent.com/u/1451688?v=4" width="72" alt="Yunlong Gao (@DPLL)" title="Yunlong Gao (@DPLL)"></a>
</p>

Contributors are displayed in the order defined in [CONTRIBUTORS.md](./CONTRIBUTORS.md).

## License

PanelWise is available under the [Apache License 2.0](./LICENSE). Initial contributors are listed in [CONTRIBUTORS.md](./CONTRIBUTORS.md).
