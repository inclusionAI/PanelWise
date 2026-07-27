# PanelWise：历史 DRACO 实验中的分数与成本优势

## 摘要

PanelWise 是一套自管的多模型深度研究与融合管线。多个研究 agent 独立搜索和撰写报告，分析阶段比较它们的共识、矛盾与遗漏，合成阶段再生成最终结果。

本报告描述发布前保存下来的历史聚合结果。在我们当时进行的 100 任务实验中，PanelWise 前沿组合的 73.68 高于当时记录的 OpenRouter Fusion 公开分数 68.3；budget 组合的 66.42 高于当时记录的 Fable 5 65.3，内部记录成本约为外部成本估计的 1/5。

这些是对**特定历史实验与当时公开参考值**的结果声明，不是对当前版本、任意配置或严格协议等价性的普遍保证。

## 管线

```text
多个研究 agent 并行研究同一问题
        ↓
分析阶段提取共识、矛盾、独特发现和盲区
        ↓
合成器生成最终报告
        ↓
项目本地 judge 或可选 official rubric grader 评分
```

研究 agent 可以使用搜索，并在 `RESEARCH_ENABLE_DIRECT_FETCH=1` 时使用本地 `httpx` 抓取网页。当前代码默认启用 direct fetch，但历史聚合文件没有记录该设置，因此不能推断历史运行使用了当前默认值。

## 保存的历史聚合结果

以下结果来自 100 个任务的发布前实验。数字直接取自仓库中的 JSON 聚合文件：

| 历史配置 | DRACO 聚合分数 | 记录成本 | 聚合证据 |
|---|---:|---:|---|
| 前沿组合 Fusion（Opus 4.8 + GPT-5.5 + Gemini 3.1 Pro），文件标记为 official grader | 73.68 | 生成与本地评分运行记录 $401.31（约 $4.01/题）；official regrade 另记录 $47.92 | [`frontier_fusion_official.json`](./results/frontier_fusion_official.json)、[`frontier_fusion_ourjudge.json`](./results/frontier_fusion_ourjudge.json) |
| 同一 official-grader 聚合中的 Claude Opus 4.8 单模型结果 | 64.57 | 未单独记录 | [`frontier_fusion_official.json`](./results/frontier_fusion_official.json) |
| GLM 5.1 + MiniMax M3 + Qwen 3.7 Max 的 Budget Fusion | 66.42 | $130.84（约 $1.31/题） | [`budget_fusion_glm_minimax_qwen.json`](./results/budget_fusion_glm_minimax_qwen.json) |
| Budget panel 替换为 Gemini 3.5 Flash 后的 Fusion | 70.95 | $80.08 增量成本；复用了已有 panel 报告 | [`budget_swap_geminiflash.json`](./results/budget_swap_geminiflash.json) |

其他已保存的历史聚合包括：

- [`results/frontier_fusion_ourjudge.json`](./results/frontier_fusion_ourjudge.json)：项目本地 judge 的前沿组合聚合。
- [`results/budget_swap_deepseek.json`](./results/budget_swap_deepseek.json)：DeepSeek panel swap 聚合。
- [`results/solo_ranking.json`](./results/solo_ranking.json)：单模型子集排名聚合。
- [`results/synthesizer_ranking.json`](./results/synthesizer_ranking.json)：合成器子集排名聚合。

这些文件只有模型标识、分数、样本数、部分领域分数、耗时或成本等聚合字段，不包含题目、rubric、答案、逐题报告、消息或轨迹。

## 证据限制

现有聚合文件没有构成完整、不可变的运行 manifest。它们未完整记录：

- 生成代码的确切提交和 dirty state；
- Python 与全部依赖的锁定版本；
- provider 侧的确切模型快照；
- 搜索后端、direct-fetch 和域名屏蔽设置；
- 提示词版本、重试、失败处理和并发参数；
- 数据集内容哈希和 grader 的完整运行标识；
- 各阶段统一口径的生成、搜索、分析、合成和评分成本。

因此，不能用当前代码默认值补写历史配置，也不能从这些聚合文件独立重建或审计完整实验。仓库有意不发布原始 DRACO 题目、rubric、任务答案、完整模型报告或轨迹。

## 历史比较结论

早期内部报告记录过 OpenRouter Fusion **68.3** 和 Fable 5 **65.3**，以及 Fable 5 约 **$7/题**的成本估计。这些数字的原始证据、协议和成本边界没有保存在本仓库中。

按我们当时采用的比较口径：

1. 前沿组合 73.68 比 OpenRouter Fusion 68.3 高 **5.38 分**。
2. Budget Fusion 66.42 比 Fable 5 65.3 高 **1.12 分**。
3. Budget Fusion 的内部记录成本约 $1.31/题，是当时 Fable 5 外部成本估计 ~$7/题的约 **19%**，即约 **1/5**。
4. Gemini 3.5 Flash panel swap 得到 70.95，但它复用了已有报告；记录的 $80.08 是增量成本，不能当作完整独立运行的端到端成本。

因此，PanelWise 保留“在该次历史实验中超过当时公开参考分数并显示成本优势”的结论。该结论不声称协议已被证明完全等价：数据集版本、模型快照、搜索工具、污染控制、grader、重试、失败处理和成本边界任一差异都可能影响比较。

## 如何理解这些结果

历史聚合支持上述特定、带时间范围的比较声明，也说明这套管线曾在若干配置下完成 100 任务运行。它们不能单独证明多模型融合在所有条件下普遍优于单模型，也不能证明某种 panel 选择具有可迁移的因果优势。

若未来需要把声明升级为当前、可重复、协议等价或更普遍的产品结论，应先完成版本化 benchmark 协议、双侧污染控制、不可变运行 manifest、可审计聚合、统计分析和外部 baseline evidence card。这部分属于发布计划中的 Gate B，不是初始源码发布的前置条件。

## 项目范围

PanelWise v0.1 面向可信用户在自己控制的机器、凭据和网络上运行研究代码。它不是面向不可信输入的托管服务。受支持、实验性和归档脚本的划分见 [`README.md`](./README.md#v01-支持范围)。

项目使用 [Apache License 2.0](./LICENSE)，初始贡献者见 [`CONTRIBUTORS.md`](./CONTRIBUTORS.md)，CI 状态见 [GitHub Actions](https://github.com/inclusionAI/PanelWise/actions/workflows/ci.yml)。
