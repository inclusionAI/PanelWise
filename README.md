# PanelWise — 在历史 DRACO 实验中复现并超过 OpenRouter Fusion 公开分数

[![CI](https://github.com/inclusionAI/PanelWise/actions/workflows/ci.yml/badge.svg)](https://github.com/inclusionAI/PanelWise/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

PanelWise 是一套自管的多模型深度研究系统：多个研究 agent 独立检索和撰写报告，随后由分析与合成阶段生成最终结果。在我们当时进行的 100 任务历史实验中，前沿组合得到 **73.68**，高于当时记录的 OpenRouter Fusion 公开分数 **68.3**；budget 组合得到 **66.42**，高于当时记录的 Fable 5 **65.3**，内部记录成本约为其公开成本估计的 **1/5**。

> 完整、面向非技术读者的介绍见 [`FUSION_REPORT.md`](./FUSION_REPORT.md)。

## 历史实验中的优势结果

以下结论描述我们在当时实验中实际记录的结果。它们支持“该次 PanelWise 实验超过当时公开参考分数、budget 运行具有记录成本优势”的历史声明，但不代表所有配置、时间或协议下都成立。

| 历史配置 | DRACO 聚合分数 | 记录成本 | 证据 |
|---|---:|---:|---|
| 前沿组合 Fusion（Opus 4.8 + GPT-5.5 + Gemini 3.1 Pro），聚合文件标记为 official grader | 73.68 | 生成与本地评分运行记录 $401.31（约 $4.01/题）；official regrade 另记录 $47.92 | [`frontier_fusion_official.json`](./results/frontier_fusion_official.json)、[`frontier_fusion_ourjudge.json`](./results/frontier_fusion_ourjudge.json) |
| 同一 official-grader 聚合中的 Claude Opus 4.8 单模型结果 | 64.57 | 未单独记录 | [`frontier_fusion_official.json`](./results/frontier_fusion_official.json) |
| GLM 5.1 + MiniMax M3 + Qwen 3.7 Max 的 Budget Fusion | 66.42 | $130.84（约 $1.31/题） | [`budget_fusion_glm_minimax_qwen.json`](./results/budget_fusion_glm_minimax_qwen.json) |
| Budget panel 替换为 Gemini 3.5 Flash 后的 Fusion | 70.95 | $80.08 增量成本；该实验复用了已有 panel 报告，不是完整独立运行成本 | [`budget_swap_geminiflash.json`](./results/budget_swap_geminiflash.json) |

- **历史分数优势：**73.68 比当时记录的 OpenRouter Fusion 68.3 高 **5.38 分**。
- **历史 budget 优势：**66.42 比当时记录的 Fable 5 65.3 高 **1.12 分**；约 $1.31/题相当于当时外部成本估计 ~$7/题的约 **19%**。
- **额外结果：**复用已有报告的 Gemini 3.5 Flash panel swap 得到 70.95，但不能把它的 $80.08 增量成本表述成完整端到端成本。

这些比较采用当时公开数字作为参考，但本仓库没有保存足以证明协议完全等价的外部证据。数据集版本、模型快照、搜索设置、grader、重试和成本口径可能存在差异。因此这里的“超过”和“成本优势”严格限定于**我们当时记录的实验结果与当时记录的公开参考值**，不是对当前产品表现或协议等价性的普遍声明。

仓库不包含原始题目、rubric、逐题报告、轨迹或完整运行 manifest。现有产物能够确认模型名称、样本数、聚合分数以及部分领域/成本字段；无法确认的设置包括确切模型快照、代码提交、依赖锁、搜索后端、direct-fetch、域名屏蔽、提示词版本和失败处理。未知设置保持标记为未知，不从当前默认值倒推。详见 [`FUSION_REPORT.md`](./FUSION_REPORT.md)。

## 管线

```
自建研究 agent（ReAct 循环 + Exa 搜索 + 全 trajectory + 无黑盒上限）
  ×N 并行研究同一题
      → fusion judge 结构化分析（共识/矛盾/独特发现/盲区）
      → synthesizer 合成终稿
      → DRACO 官方 grader 评分
```

## 仓库结构

```
draco_eval/            核心包
  dataset.py           DRACO 数据集加载 + 加权 rubric 解析
  openrouter.py        异步 OpenRouter 客户端（限速/重试/成本）
  research_agent.py    自建 ReAct 研究 agent（Exa 搜索 + web_fetch + trajectory）
  fusion.py            Fusion 管线（panel → judge 分析 → synthesizer）
  judge.py             我们的 DRACO 裁判
  official_judge.py    接官方 rubric grader（paper-instruments/rubric）
  scoring.py           DRACO 评分公式
  agent.py / pipeline.py / run.py
fusion_full.py         前沿组合全量评测
budget_fusion_full.py  便宜组合全量评测
panel_swap.py          换 panel 第三人（复用前两位报告）
solo_ranking.py        单模型排名
synth_ablation.py / budget_fusion_synth.py   合成器选型
*_ablation.py / compare_*.py / regrade_official.py   各项对照实验
results/               历史聚合结果（json）
FUSION_REPORT.md       历史实验与证据限制说明
```

## v0.1 支持范围

### 受支持接口

- `draco_eval` 核心包中明确写入 README 的接口；未记录的接口不承诺兼容性。
- `draco_eval.fusion.run_fusion_messages` 和 `draco_eval.research_agent.run_research_messages`。
- `fusion_full.py` 的 `--dry-run`、有界 `--limit 1` 和由用户明确发起的完整运行路径。
- README 中记录的 direct-fetch 配置、本地 judge 和可选 official grader 安装路径。

以下划线开头的辅助函数（包括 `_analyze` 和 `_synthesize`）属于内部实现，不是稳定 API。

### 实验性、best-effort 工作流

`fusion_demo.py`、`research_demo.py`、`budget_fusion_full.py`、`budget_fusion_synth.py`、`solo_ranking.py` 和 `regrade_official.py` 作为研究工作流保留。它们可能依赖特定模型、可选依赖、付费服务或先前生成的中间产物，不享有与受支持接口相同的兼容性承诺。

### 归档研究脚本

`aggregate_sweep.py`、`backend_ablation.py`、`compare_modes.py`、`compare_selfbuilt_vs_st.py`、`engine_compare.py`、`judge_delta.py`、`native_retest.py`、`or_fusion_compare.py`、`panel_swap.py`、`rejudge.py`、`rerun_failed.py`、`search_tier_ablation.py` 和 `synth_ablation.py` 为研究透明度而保留，但不是 PanelWise v0.1 的稳定接口。这些脚本可能依赖未随仓库发布的历史模型、配置或中间产物。

## 运行

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # 填 OPENROUTER_API_KEY；搜索用 Exa 需 EXA_API_KEY

# 首先执行无网络、无付费调用的命令检查配置和入口
.venv/bin/python fusion_full.py --dry-run

# 有意进行一个有上限的冒烟运行（会产生模型和搜索费用）
.venv/bin/python fusion_full.py --limit 1
```

`fusion_full.py --dry-run` 不加载数据、不创建输出、也不调用模型、搜索或 grader。`--limit 1`
明确限制为一个任务；它不是免费的 dry run。

> **全量运行会产生费用。** 下列命令会进行付费模型调用；自管 research agent 使用 Exa 时还会
> 产生 Exa 搜索费用。先运行 dry run，再用 `--limit 1` 验证你的 key、模型和预算。

```bash
# 前沿组合全量（100 题；存答案版，使用项目本地 judge）
.venv/bin/python fusion_full.py

# 便宜组合全量（100 题；使用官方 rubric grader）
# 先安装可选的官方 grader 依赖：.venv/bin/pip install -r requirements-eval.txt
AGENT_REASONING_EFFORT=high .venv/bin/python budget_fusion_full.py

# 单模型排名（10 子集；同样会产生模型/搜索费用）
.venv/bin/python solo_ranking.py
```

上述完整评测工作流支持将记录写入 `output/` 后恢复运行；实验性和归档脚本的恢复行为取决于各自实现，不作统一保证。`output/` 已被 gitignore。

### 安装、grader 与运行环境

- 推荐并已验证的完整路径是 Python 3.11。核心依赖当前也可在 Python 3.9 上运行；可选的
  `rubric==2.2.0` 要求 Python 3.10+（其发布页面声明支持 3.10–3.13）。
- `requirements.txt` 仅包含核心运行依赖。`requirements-eval.txt` 是可选的官方 grader 依赖，
  固定为 `rubric==2.2.0`；其 canonical source 是
  [`paper-instruments/rubric`](https://github.com/paper-instruments/rubric)。
- `fusion_full.py` 使用本仓库的 `draco_eval.judge.judge_once`，不是 `rubric`。
  `budget_fusion_full.py`、`budget_fusion_synth.py`、`solo_ranking.py`、`regrade_official.py`
  以及其他 official-grader experiment scripts 都导入 `draco_eval.official_judge`，因此需要
  先安装 `requirements-eval.txt`。
- `fusion_full.py` 是 provider-flexible 的受支持入口：默认 OpenRouter 需要
  `OPENROUTER_API_KEY`；`MODEL_PROVIDER=zenmux` 时需要 `ZENMUX_API_KEY`，且 ZenMux 仅支持
  `RESEARCH_SEARCH_BACKEND=exa`。OpenRouter 支持 `exa`、`native` 和 `exa_or`；直连 Exa 搜索
  需要 `EXA_API_KEY`，`native`/`exa_or` 使用 OpenRouter server tools。`budget_fusion_full.py`、
  `budget_fusion_synth.py`、`solo_ranking.py` 和 `regrade_official.py` 是 OpenRouter-only
  historical/evaluation scripts，均需要 `OPENROUTER_API_KEY`。
- 运行产生的可忽略输出位于 `output/fusion_full_v2*/`、`output/budget_full/`、
  `output/budget_panel/`、`output/budget_synth/` 和 `output/fusion_full_v2/official_grades/`。
  不要将报告、下载数据、日志、`.env` 或凭据提交到 Git。
  若切换 direct-fetch 模式，请设置 `FUSION_OUT`、`BUDGET_PANEL_DIR`、`BUDGET_FULL_REC_DIR`、
  `BUDGET_SYNTH_DIR` 或 `SOLO_RANKING_OUT` 到新的输出目录，避免复用不同模式的缓存。

### 自管研究 agent 的本地抓取

`RESEARCH_ENABLE_DIRECT_FETCH=1` 是默认设置，保留自管 research agent 的 Exa 搜索加本地
`httpx` 页面抓取行为。若运行环境不应允许本地进程抓取任意模型请求的页面，请在 `.env`
中设为 `RESEARCH_ENABLE_DIRECT_FETCH=0`；此时该 agent 只会获得 `web_search`，研究质量可能
因无法读取完整原始页面而下降。

启用本地抓取时，PanelWise 保留现有的本地网络访问风险；该开关只是可信本地运行场景的
opt-out，并非 SSRF 防护。它不影响 OpenRouter provider-hosted 的 `web_fetch` 工具：后者由
OpenRouter 执行，且不受此本地设置控制。

## Python 输入接口

内部调用既可使用原来的 prompt 字符串，也可传入 provider-compatible 的消息数组：

```python
from draco_eval.fusion import run_fusion_messages
from draco_eval.research_agent import run_research_messages

report = await run_research_messages(client, "研究这个问题", "model", [], request_id="request-1")

fusion = await run_fusion_messages(
    client,
    [
        {"role": "system", "content": "Use concise citations."},
        {"role": "user", "content": [{"type": "input_text", "text": "研究这个问题"}]},
    ],
    panel_models=["model-a", "model-b"],
    synth_model="model-c",
    judge_model="model-d",
    excluded_domains=[],
    request_id="request-1",
)
```

字符串会转换成一个 user message；消息数组会被保留其顺序、字段和结构化 content，且不会修改调用方对象。空数组会在本地报错。角色、tool history、模态、上下文大小及每个模型是否支持这些消息，仍完全由 provider 和所选模型决定。

## 依赖

- OpenRouter（统一调各家模型）
- Exa（深度研究的网页搜索原语）
- DRACO 数据集 + 官方 rubric grader（评分对齐官方口径）

## 许可证与贡献者

PanelWise 使用 [Apache License 2.0](./LICENSE)。初始贡献者见 [`CONTRIBUTORS.md`](./CONTRIBUTORS.md)。
