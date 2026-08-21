<p align="center">
  <img src="./assets/panelwise-logo.svg" alt="PanelWise logo" width="720">
</p>

<p align="center">
  <a href="./README.md">English</a> · <strong>中文</strong>
</p>

<p align="center">
  <a href="https://github.com/inclusionAI/PanelWise/actions/workflows/ci.yml"><img src="https://github.com/inclusionAI/PanelWise/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache 2.0 license"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-273D5D.svg" alt="Python 3.10 或更高版本">
</p>

# PanelWise

**融合多个模型彼此互补的优势，得到一个更强的结果。**

PanelWise 的出发点是一个简单的问题：如果每个模型的判断中都有值得保留的部分，能否融合这些互补判断，得到超过任一单独回答的结果？PanelWise 是一套可嵌入的 Python 多模型聚合引擎：它把同一项任务交给可配置的模型 panel，保留每个模型的独特贡献，再通过两种执行拓扑之一协调这些贡献。任务类型本身不受限制；prompt、provider 和环境适配器决定具体领域。

<p align="center">
  <img src="./assets/panelwise-flow-concept.png" alt="三个不同模型的视角经过 PanelWise 后融合为一个答案" width="1000">
</p>

## 一个接口，两种模式

| 模式 | 执行拓扑 | 适合场景 |
|---|---|---|
| `--eval` | 独立完整作答 → evaluator → 最终融合 | 以答案为中心的任务，包括深度研究 |
| `--no-eval` | 独立提出下一步 → 协调为一个行动 → 观察共享状态 → 继续 | 有状态任务，包括代码修改 |

它们是执行模式，不是写死的任务分类。两种模式都接受任意任务字符串；自定义 `ChatClient` 和 `Executor` 可以让同一引擎连接其他模型网关或执行环境。

```bash
panelwise run "比较 PostgreSQL 和 MySQL 在大型电商平台中的取舍" --eval
panelwise run "修复 parser 回归并运行相关测试" --no-eval --workspace ./project
```

`--eval` 与 `--no-eval` 只覆盖 YAML 中的一个配置值，因此应用无需改代码或复制配置就能切换拓扑。

## 安装

PanelWise 要求 Python 3.10 或更高版本。

```bash
git clone https://github.com/inclusionAI/PanelWise.git
cd PanelWise
python3 -m venv .venv
.venv/bin/python -m pip install .
```

也可以直接安装指定 Git 版本：

```bash
python3 -m pip install "panelwise @ git+https://github.com/inclusionAI/PanelWise.git@main"
```

需要可复现安装时，请把 `main` 替换为 release tag 或 commit SHA。参与开发请使用 `python -m pip install -e '.[dev]'`。

## 五分钟开始

先创建带完整注释的配置，并在不调用模型的情况下验证：

```bash
panelwise init
export OPENROUTER_API_KEY="..."
panelwise validate --config panelwise.yaml
```

生成的 YAML 包含 provider、模型角色、并发、超时、workspace 和 prompt 设置。同一个文件控制两种模式：

```yaml
version: 1

provider:
  name: openrouter
  api_key_env: OPENROUTER_API_KEY

models:
  panel: [z-ai/glm-5.1, minimax/minimax-m3, qwen/qwen3.7-max]
  coordinator: z-ai/glm-5.1
  evaluator: z-ai/glm-5.1

execution:
  eval: true
  concurrency: 3
  max_steps: 24
  workspace: .
```

完整配置见 [`panelwise.example.yaml`](./panelwise.example.yaml) 与[配置说明](./docs/configuration.md)。

## Python Library

```python
import asyncio
from panelwise import PanelWise

async def main():
    async with PanelWise.from_yaml("panelwise.yaml", eval=True) as engine:
        result = await engine.run(
            "比较碳税最有力的支持与反对观点。",
            request_id="example-1",
        )

    if result.ok:
        print(result.output)
    else:
        print(result.status, result.errors)

asyncio.run(main())
```

`PanelWiseResult` 在两种模式下使用同一个稳定契约，始终包含 `request_id`、`mode`、`status`、`output`、`usage` 和 `errors`。Eval 模式额外返回独立 `candidates` 与结构化 `evaluation`；no-eval 模式返回共享 `trajectory` 和 executor `artifacts`，在可用时包括 Git patch。

单个 panel 成员失败不会中断其他成员。无效配置会抛出明确的 `PanelWiseError` 子类，不会用 `SystemExit` 终止宿主进程。详见 [Python API 与失败契约](./docs/api.md)。

## DRACO 结果

我们在 [DRACO](https://arxiv.org/abs/2503.14476) 的全部 100 个任务上评测了 PanelWise。DRACO 是一个覆盖十个领域的深度研究 benchmark，每道题通过加权 rubric 评估事实准确性、广度、深度、表达和引用质量。

<p align="center">
  <img src="./assets/panelwise-draco-results.png" alt="PanelWise DRACO 跑分与 OpenRouter Fusion、Claude Fable 5 公开结果的比较" width="1000">
</p>

其中 **OpenRouter Fusion 68.3 分**与 **Claude Fable 5 65.3 分**均引用自 [OpenRouter 官方 Fusion 发布文章](https://openrouter.ai/blog/announcements/fusion-beats-frontier/)。PanelWise 分数来自我们在全部 100 个任务上的完整评测运行。

| 系统 | 模型 | DRACO 分数 |
|---|---|---:|
| **PanelWise 前沿组合** | Opus 4.8 + GPT-5.5 + Gemini 3.1 Pro | **73.68** |
| OpenRouter Fusion 公开结果 | Opus 4.8 + GPT-5.5 + Gemini 3.1 Pro | 68.3 |
| **PanelWise 轻量组合** | GLM 5.1 + MiniMax M3 + Qwen 3.7 Max | **66.42** |
| 公开结果中最强的单模型基线 | Claude Fable 5 | 65.3 |

前沿组合比同三个参与模型的 Fusion 公开结果高 **5.38 分**；完整独立运行的轻量组合比公开结果中最强的单模型基线高 **1.12 分**。OpenRouter 的 Fable 5 结果只覆盖实际完成的 93 个任务，另有 7 个任务被内容过滤器拦截，因此该比较存在轻微不对称。

这项 benchmark 支持的是 PanelWise 在 DRACO 深度研究任务上的聚合效果，并不意味着任意任务或任意模型组合都必然提升。

## Provider 与扩展接口

内置异步客户端支持：

- OpenRouter；
- ZenMux；
- 任意 OpenAI-compatible chat-completions 地址。

Provider 专用请求字段可以直接写入 YAML。非 OpenAI-compatible provider 只需实现很小的 `ChatClient` 协议。自定义共享环境实现 `Executor` 即可，让 no-eval 模式协调 shell、容器、浏览器、数据库、模拟器或远程 worker，而不修改聚合引擎。

Trajectory 模式的内置本地 shell executor 会拒绝明显的破坏性命令、限制运行时间与输出，并从一个配置好的 workspace 启动每项操作。它只是防护栏，不是 sandbox；不可信任务应运行在一次性容器或虚拟机里。

## 文档

- [CLI 说明](./docs/cli.md)
- [配置与 provider 设置](./docs/configuration.md)
- [Python API 与结果契约](./docs/api.md)
- [示例](./examples)
- [故障排查](./docs/troubleshooting.md)
- [发布流程](./docs/releasing.md)
- [贡献与 PR 要求](./CONTRIBUTING.md)
- [安全策略](./SECURITY.md)
- [Changelog](./CHANGELOG.md)

## 开发

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff check src tests examples
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m build
```

测试完全离线，不需要模型 key。公共 API、provider、executor 与架构变更应先创建 issue；集中、明确的文档修正可以直接发 PR。每项行为变更都必须附可复现验证。详见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 贡献者

<p>
  <a href="https://github.com/jcguo123"><img src="https://avatars.githubusercontent.com/u/164945525?v=4" width="72" alt="Jiacheng Guo (@jcguo123)" title="Jiacheng Guo (@jcguo123)"></a>
  <a href="https://github.com/DPLL"><img src="https://avatars.githubusercontent.com/u/1451688?v=4" width="72" alt="Yunlong Gao (@DPLL)" title="Yunlong Gao (@DPLL)"></a>
</p>

贡献者按照 [CONTRIBUTORS.md](./CONTRIBUTORS.md) 中约定的顺序展示。

## 许可证

PanelWise 使用 [Apache License 2.0](./LICENSE)。
