"""集中配置：模型、并发、API 端点、防污染域名。

绝大多数旋钮都能用环境变量或 CLI 覆盖，方便在不改代码的情况下调参。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv 没装也能跑，只是不会自动读 .env
    pass


# ---- Provider selection ----
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter").strip().lower()


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        raw = default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ---- OpenRouter ----
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# 友好头：OpenRouter 建议带上，方便它做用量归因（可留空）
OR_REFERER = os.getenv("OPENROUTER_REFERER", "https://github.com/inclusionAI/PanelWise")
OR_TITLE = os.getenv("OPENROUTER_TITLE", "PanelWise")

# ---- ZenMux ----
ZENMUX_BASE_URL = os.getenv("ZENMUX_BASE_URL", "https://zenmux.ai/api/v1")
ZENMUX_API_KEY = os.getenv("ZENMUX_API_KEY", "")
ZENMUX_MODEL_MAP = os.getenv("ZENMUX_MODEL_MAP", "{}")
ZENMUX_JSON_SCHEMA_FALLBACK = _env_flag("ZENMUX_JSON_SCHEMA_FALLBACK", "1")

# ---- HuggingFace 数据集 ----
DRACO_JSONL_URL = (
    "https://huggingface.co/datasets/perplexity-ai/draco/resolve/main/test.jsonl"
)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ---- 自建 server-tool（ReAct 研究 agent）----
# 搜索后端：Exa（OpenRouter/Fusion 底层同款，最忠实）。web_fetch 无 key 直接 httpx 抓。
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
EXA_BASE_URL = "https://api.exa.ai"
RESEARCH_MAX_STEPS = int(os.getenv("RESEARCH_MAX_STEPS", "20"))   # 自管循环步数上限（我们说了算，无黑盒）
RESEARCH_MAX_CONSEC_FAILS = int(os.getenv("RESEARCH_MAX_CONSEC_FAILS", "3"))  # 模型调用连续失败几次才放弃循环（之上还有 chat 内部重试）
RESEARCH_SEARCH_RESULTS = int(os.getenv("RESEARCH_SEARCH_RESULTS", "5"))
EXA_SEARCH_TYPE = os.getenv("EXA_SEARCH_TYPE", "auto")  # auto / fast / neural / keyword（Exa 档位）
# 搜索后端：
#   exa      = 直连 Exa /search（最便宜 $0.02/搜，但要单独 Exa key+充值）
#   native   = 经 OpenRouter 做一次 engine=native 搜索调用（单 key，~$0.2/搜）
#   exa_or   = 经 OpenRouter 做一次 engine=exa 搜索调用（单 key，Exa 质量，~$0.2/搜，无需 Exa 账号）
RESEARCH_SEARCH_BACKEND = os.getenv("RESEARCH_SEARCH_BACKEND", "exa")  # exa | native | exa_or
# Local httpx page fetch for the self-managed research agent. This remains enabled
# by default to preserve historical behavior; set to 0 for search-only research.
RESEARCH_ENABLE_DIRECT_FETCH = _env_flag("RESEARCH_ENABLE_DIRECT_FETCH", "1")
# via-OpenRouter 那次搜索调用用哪个模型当"包装层"。空=用 agent 自己的模型；设便宜模型(如 gemini flash)可大幅省钱。
NATIVE_SEARCH_MODEL = os.getenv("NATIVE_SEARCH_MODEL", "")
RESEARCH_FETCH_MAX_CHARS = int(os.getenv("RESEARCH_FETCH_MAX_CHARS", "8000"))
# 某些站点要求自动访问使用可识别的 User-Agent；环境变量可按需覆盖。
FETCH_USER_AGENT = os.getenv(
    "FETCH_USER_AGENT", "PanelWise/0.1 (+https://github.com/inclusionAI/PanelWise)"
)

# ---- 模型 slug（去 openrouter.ai/models 核对确切名字）----
# 被测系统（system under test）。本次默认 DeepSeek V4 Pro。
DEFAULT_AGENT_MODEL = os.getenv("AGENT_MODEL", "deepseek/deepseek-v4-pro")
# 裁判。论文用 Gemini-3-Pro；这里走 OpenRouter，单 key 即可。
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "google/gemini-3.1-pro-preview")

# ---- 评测协议 ----
DEFAULT_JUDGE_RUNS = int(os.getenv("JUDGE_RUNS", "5"))  # 论文：每题独立打分 5 次

# ---- 联网（OpenRouter 自带）----
# server_tool: tools:[{type:"openrouter:web_search"},{type:"openrouter:web_fetch"}]
#              —— 服务端 agentic 多步循环，模型自己决定搜/抓几次（对齐 Fusion / 论文）。
# plugin:      plugins:[{id:"web"}] —— 旧的单轮 RAG（已被 OpenRouter 废弃，保留作回退）。
AGENT_WEB_MODE = os.getenv("AGENT_WEB_MODE", "server_tool")
WEB_ENGINE = os.getenv("WEB_ENGINE", "exa")
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "5"))      # 每次搜索返回条数
# 0 = 不传 = agentic 循环不设累计上限（让模型搜够为止；同 max_tokens 的教训：
# 人为掐断会让难题写不完报告）。要控成本可设正数。
WEB_MAX_TOTAL_RESULTS = int(os.getenv("WEB_MAX_TOTAL_RESULTS", "0"))
WEB_CONTEXT_SIZE = os.getenv("WEB_CONTEXT_SIZE", "high")      # low|medium|high 每条摘录上下文预算

# ---- 防数据污染 ----
# panel/agent 联网时屏蔽掉托管 DRACO rubric / 答案的域名，否则模型会搜到评分标准。
BLOCKED_DOMAINS = [
    "huggingface.co",
    "r2cdn.perplexity.ai",
    "arxiv.org",
    "github.com",  # 含开源 grader 仓库
    "perplexity.ai",
]
ENFORCE_BLOCKED_FETCH = _env_flag("ENFORCE_BLOCKED_FETCH", "0")

# ---- 并发与网络 ----
# task 级并发：默认等于子集大小（10/100），即“一次跑完”。CLI 可覆盖。
# HTTP 级并发：限制同时在飞的请求总数，避免触发 OpenRouter 限速。
DEFAULT_MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "40"))

HTTP_CONNECT_TIMEOUT = 30.0
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "900"))  # 深度研究可能跑几分钟
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "4"))

# max_tokens：默认 0 = 不传，让 provider 用模型自己的满额上限。
# 教训：server-tool agentic 模式下，单次响应总输出（工具编排+推理+最终报告）不可预测，
# 设任何上限都可能把最终报告截没 → 出空/稀薄报告。除非明确要控成本，否则别设。
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "0"))
# 只拦"空/垃圾"输出（如把问题原样吐回 ~270 字）的小地板，触发重试；
# 不丢弃达不到的报告（仍返回最长那篇并标 warn），所以不会压分。设很低即可。
# 注意：不要设 0——否则空报告会静默判 0 而非 agent_failed，丢失诊断信号。
AGENT_MIN_REPORT_CHARS = int(os.getenv("AGENT_MIN_REPORT_CHARS", "300"))
AGENT_MAX_ATTEMPTS = int(os.getenv("AGENT_MAX_ATTEMPTS", "3"))
# reasoning effort：拉满设 "high"，空=不传
AGENT_REASONING_EFFORT = os.getenv("AGENT_REASONING_EFFORT", "")
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "0"))  # 同理默认不传；太小会截断 grades JSON


def validate_provider_config(dry_run: bool = False) -> None:
    """Validate provider-specific API keys and incompatible backend choices."""
    provider = MODEL_PROVIDER.strip().lower()
    if provider not in {"openrouter", "zenmux"}:
        raise SystemExit(
            f"不支持的 MODEL_PROVIDER={MODEL_PROVIDER!r}；请使用 openrouter 或 zenmux。"
        )
    if provider == "zenmux" and RESEARCH_SEARCH_BACKEND in {"native", "exa_or"}:
        raise SystemExit(
            "MODEL_PROVIDER=zenmux 不支持 RESEARCH_SEARCH_BACKEND=native 或 exa_or；"
            "这两个后端依赖 OpenRouter server tools。请设置 RESEARCH_SEARCH_BACKEND=exa。"
        )
    if dry_run:
        return
    if provider == "openrouter" and not OPENROUTER_API_KEY:
        raise SystemExit(
            "缺少 OPENROUTER_API_KEY。复制 .env.example 为 .env 填入，或 export OPENROUTER_API_KEY=..."
        )
    if provider == "zenmux" and not ZENMUX_API_KEY:
        raise SystemExit(
            "缺少 ZENMUX_API_KEY。请在 .env 中填写，或 export ZENMUX_API_KEY=..."
        )


@dataclass
class RunConfig:
    """单次运行的有效配置（由 CLI 组装）。"""

    subset: int = 10              # 10 / 100 / 任意 N
    concurrency: int = 10         # task 级并发
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    agent_model: str = DEFAULT_AGENT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_runs: int = DEFAULT_JUDGE_RUNS
    out_dir: str = "output"
    no_judge: bool = False        # 只跑生成、不打分（省裁判钱）
    dry_run: bool = False         # 只加载数据 + 打印选择，不调 API
    blocked_domains: list[str] = field(default_factory=lambda: list(BLOCKED_DOMAINS))

    def validate(self) -> None:
        validate_provider_config(dry_run=self.dry_run)
