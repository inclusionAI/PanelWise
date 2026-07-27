"""自建 server-tool：自管 ReAct 循环的深度研究 agent，全程 trajectory 可见。

对标 OpenRouter server-tool，但：
- 循环跑在我们这边 → 每一步 tool_call / 搜索 query / 抓取的网页 / 推理全部留痕。
- 无黑盒 max_tool_calls=8 枷锁；步数上限我们说了算。
- 循环结束强制收尾写报告 → 根治"研究上瘾不交活"（Finance 那种）。

工具用标准 function-calling（不是 openrouter: server tool），由我们解析 tool_calls 并执行：
- web_search  -> Exa /search（需要 EXA_API_KEY）
- web_fetch   -> 直接 httpx GET + 去标签（无需 key）
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx

from . import config
from .dataset import Task
from .clients import ChatClient
from .messages import ModelInput, normalize_messages
from .openrouter import message_text

try:
    import trafilatura  # 主正文抽取（去导航/广告/噪声）
except Exception:  # noqa: BLE001
    trafilatura = None
try:
    from pypdf import PdfReader  # PDF 文本提取
except Exception:  # noqa: BLE001
    PdfReader = None

RESEARCH_SYSTEM_PROMPT = (
    "You are an expert deep-research agent. You have two tools: web_search (find sources) "
    "and web_fetch (read a specific URL in full).\n"
    "Work iteratively: search → read primary sources → refine queries → verify key facts. "
    "Use multiple searches and fetch the most authoritative pages before writing.\n"
    "When you have gathered enough, STOP calling tools and write a comprehensive, accurate, "
    "well-structured final report that fully answers the task.\n"
    "- Cite sources inline with URLs. Be precise with numbers, dates, entities, units.\n"
    "- Address every sub-question, persona, comparison, and constraint in the task.\n"
    "- Never fabricate. If a figure can't be found, say so explicitly.\n"
    "- Do NOT consult or cite benchmark/rubric pages; answer on the merits."
)

RESEARCH_SEARCH_ONLY_SYSTEM_PROMPT = (
    "You are an expert deep-research agent with a web_search tool for finding sources.\n"
    "Work iteratively: search for primary sources, refine queries, and verify key facts from "
    "the search results. Use multiple targeted searches before writing.\n"
    "When you have gathered enough, STOP calling tools and write a comprehensive, accurate, "
    "well-structured final report that fully answers the task.\n"
    "- Cite sources inline with URLs. Be precise with numbers, dates, entities, units.\n"
    "- Address every sub-question, persona, comparison, and constraint in the task.\n"
    "- Never fabricate. If a figure can't be found, say so explicitly.\n"
    "- Do NOT consult or cite benchmark/rubric pages; answer on the merits."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for sources. Returns titles, URLs, and excerpts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the full text content of a specific URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
]

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")
_WEB_FETCH_MAX_REDIRECTS = 20


@dataclass
class TrajectoryStep:
    step: int
    kind: str           # "assistant" | "tool_result"
    text: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    result_preview: str = ""
    error: str = ""


@dataclass
class ResearchResult:
    task_id: str
    report: str
    ok: bool
    trajectory: list[TrajectoryStep]
    n_searches: int
    n_fetches: int
    n_steps: int
    error: str = ""

    def trajectory_dicts(self) -> list[dict]:
        return [vars(s) for s in self.trajectory]


async def _exa_search(query: str, excluded_domains: list[str]) -> str:
    if not config.EXA_API_KEY:
        return "ERROR: EXA_API_KEY 未设置，无法搜索。"
    body = {
        "query": query,
        "numResults": config.RESEARCH_SEARCH_RESULTS,
        "type": config.EXA_SEARCH_TYPE,
        "contents": {"highlights": True, "text": {"maxCharacters": 1200}},
    }
    if excluded_domains:
        body["excludeDomains"] = excluded_domains
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{config.EXA_BASE_URL}/search",
            headers={"x-api-key": config.EXA_API_KEY, "Content-Type": "application/json"},
            json=body,
        )
        r.raise_for_status()
        data = r.json()
    out = []
    for i, res in enumerate(data.get("results", []), 1):
        title = res.get("title", "")
        url = res.get("url", "")
        hl = res.get("highlights") or []
        snippet = " […] ".join(hl) if hl else (res.get("text", "") or "")[:600]
        out.append(f"[{i}] {title}\n    {url}\n    {snippet}")
    return "\n\n".join(out) if out else "（无结果）"


def _pdf_to_text(content: bytes) -> str:
    if PdfReader is None:
        return "ERROR: PDF 提取库未安装"
    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages[:60]:  # 限页防超长年报
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
    return _WS_RE.sub("\n\n", "\n".join(parts)).strip()


def _html_to_text(html: str, url: str) -> str:
    # 首选 trafilatura 抽主正文；失败再退回粗暴去标签
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(html, url=url, include_links=True, favor_recall=True)
            if extracted and len(extracted.strip()) > 200:
                return extracted.strip()
        except Exception:  # noqa: BLE001
            pass
    text = _TAG_RE.sub(" ", html)
    text = _HTML_RE.sub(" ", text)
    return _WS_RE.sub("\n\n", text).strip()


def _url_matches_blocked_domain(url: str, blocked_domains: list[str]) -> bool:
    # Host-string guard only; this intentionally does not resolve IP literals to domains.
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    for domain in blocked_domains:
        blocked = domain.lower().rstrip(".")
        if host == blocked or host.endswith(f".{blocked}"):
            return True
    return False


async def _web_fetch(url: str) -> str:
    if config.ENFORCE_BLOCKED_FETCH and _url_matches_blocked_domain(url, config.BLOCKED_DOMAINS):
        return "ERROR: blocked benchmark/rubric domain"
    async with httpx.AsyncClient(
        timeout=60,
        follow_redirects=not config.ENFORCE_BLOCKED_FETCH,
        max_redirects=_WEB_FETCH_MAX_REDIRECTS,
    ) as c:
        if config.ENFORCE_BLOCKED_FETCH:
            current_url = url
            for redirect_count in range(_WEB_FETCH_MAX_REDIRECTS + 1):
                r = await c.get(current_url, headers={"User-Agent": config.FETCH_USER_AGENT})
                if not r.is_redirect:
                    break
                if redirect_count == _WEB_FETCH_MAX_REDIRECTS:
                    raise httpx.TooManyRedirects(f"Exceeded {_WEB_FETCH_MAX_REDIRECTS} redirects")
                location = r.headers.get("location")
                if not location:
                    raise httpx.HTTPError("redirect response missing Location header")
                current_url = urljoin(str(r.url), location)
                if _url_matches_blocked_domain(current_url, config.BLOCKED_DOMAINS):
                    return "ERROR: blocked benchmark/rubric domain"
        else:
            r = await c.get(url, headers={"User-Agent": config.FETCH_USER_AGENT})
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        content = r.content
        response_text = r.text
    # 判 PDF：content-type 或魔数 %PDF
    if "pdf" in ctype or content[:5] == b"%PDF-":
        text = _pdf_to_text(content)
    else:
        text = _html_to_text(response_text, url)
    if not text.strip():
        return "（抓取成功但未提取到可读正文）"
    return text[: config.RESEARCH_FETCH_MAX_CHARS]


_NATIVE_SEARCH_SYS = (
    "You are a web-search tool. Search the web for the user's query and report the findings "
    "with HIGH FIDELITY — preserve specific numbers, dates, names, and short verbatim quotes "
    "from the sources; do NOT over-summarize or paraphrase figures. For each relevant source "
    "give: the source URL, then the concrete facts/excerpts found there. Cover multiple sources. "
    "Your job is to surface raw material, not to write a polished answer."
)


async def _or_search(client: ChatClient, query: str, model: str, engine: str,
                     excluded_domains: list[str]) -> str:
    """经 OpenRouter 做一次搜索调用(engine=native 或 exa)，回丰富结果。单 key，无需 Exa 账号。"""
    search_model = config.NATIVE_SEARCH_MODEL or model
    params = {
        "engine": engine,
        "max_results": config.RESEARCH_SEARCH_RESULTS,
        "search_context_size": config.WEB_CONTEXT_SIZE,
    }
    if engine == "exa" and excluded_domains:  # exa 支持域名过滤(native 不支持)
        params["excluded_domains"] = excluded_domains
    payload = {
        "model": search_model,
        "tools": [{"type": "openrouter:web_search", "parameters": params}],
        "tool_choice": "required",
        "messages": [
            {"role": "system", "content": _NATIVE_SEARCH_SYS},
            {"role": "user", "content": query},
        ],
    }
    data = await client.chat("or_search", payload)
    return message_text(data) or "（搜索未返回内容）"


async def _exec_tool(client, name: str, args: dict, model: str, excluded_domains: list[str]) -> str:
    if name == "web_fetch" and not config.RESEARCH_ENABLE_DIRECT_FETCH:
        return "ERROR: direct web_fetch is disabled by configuration"
    if name == "web_search" and config.RESEARCH_SEARCH_BACKEND in {"native", "exa_or"}:
        if not getattr(client, "supports_openrouter_server_tools", True):
            raise RuntimeError(
                f"RESEARCH_SEARCH_BACKEND={config.RESEARCH_SEARCH_BACKEND} requires "
                "OpenRouter server tools; set RESEARCH_SEARCH_BACKEND=exa for MODEL_PROVIDER=zenmux."
            )
    try:
        if name == "web_search":
            q = args.get("query", "")
            be = config.RESEARCH_SEARCH_BACKEND
            if be == "native":
                return await _or_search(client, q, model, "native", excluded_domains)
            if be == "exa_or":
                return await _or_search(client, q, model, "exa", excluded_domains)
            return await _exa_search(q, excluded_domains)
        if name == "web_fetch":
            return await _web_fetch(args.get("url", ""))
        return f"ERROR: unknown tool {name}"
    except Exception as e:  # noqa: BLE001 — 工具失败回喂给模型，让它换路
        return f"ERROR fetching/searching: {e}"


async def run_research_messages(
    client: ChatClient,
    model_input: ModelInput,
    model: str,
    excluded_domains: list[str],
    request_id: str = "",
) -> ResearchResult:
    if config.RESEARCH_SEARCH_BACKEND in {"native", "exa_or"}:
        if not getattr(client, "supports_openrouter_server_tools", True):
            raise RuntimeError(
                f"RESEARCH_SEARCH_BACKEND={config.RESEARCH_SEARCH_BACKEND} requires "
                "OpenRouter server tools; set RESEARCH_SEARCH_BACKEND=exa for MODEL_PROVIDER=zenmux."
            )
    direct_fetch_enabled = config.RESEARCH_ENABLE_DIRECT_FETCH
    system_prompt = RESEARCH_SYSTEM_PROMPT if direct_fetch_enabled else RESEARCH_SEARCH_ONLY_SYSTEM_PROMPT
    active_tools = TOOLS if direct_fetch_enabled else [TOOLS[0]]
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(normalize_messages(model_input))
    traj: list[TrajectoryStep] = []
    n_search = n_fetch = 0
    consec_fails = 0  # 连续失败计数（chat 内部已重试；这里再容忍几次，撑不住才收尾）

    for step in range(config.RESEARCH_MAX_STEPS):
        payload = {"model": model, "messages": messages, "tools": active_tools, "tool_choice": "auto"}
        if config.AGENT_REASONING_EFFORT:  # 拉满 reasoning
            payload["reasoning"] = {"effort": config.AGENT_REASONING_EFFORT}
        try:
            data = await client.chat("research", payload)
            consec_fails = 0  # 成功就清零
        except Exception as e:  # noqa: BLE001 — 中途失败：先重试这一步，连续撑不住才收尾
            consec_fails += 1
            traj.append(TrajectoryStep(step, "assistant", text="",
                                       args={"chat_error": str(e), "consec_fails": consec_fails}))
            if consec_fails >= config.RESEARCH_MAX_CONSEC_FAILS:
                break  # 连续失败太多 → 退守强制收尾，用已查到的写报告
            continue   # 否则重试本步（messages 未变，等于重发）

        msg = data["choices"][0]["message"]
        text = message_text(data)
        tool_calls = msg.get("tool_calls") or []
        traj.append(TrajectoryStep(step, "assistant", text=text,
                                   args={"tool_calls": [tc.get("function", {}).get("name") for tc in tool_calls]}))

        if not tool_calls:  # 模型停止调工具 = 给出最终报告
            # 必须够长才算真报告；空/垃圾(如只有 "}")则催它重写(防 reasoning 把正文吃掉、content 漏尾巴)
            if len(text.strip()) >= config.AGENT_MIN_REPORT_CHARS:
                return ResearchResult(request_id, text.strip(), True, traj, n_search, n_fetch, step + 1)
            messages.append({
                "role": "user",
                "content": "Your last message did not contain a usable report. Write the COMPLETE "
                           "final report now as plain prose with inline source URLs. Do not output "
                           "JSON, tool calls, or an empty/near-empty message.",
            })
            continue

        # 必须把 assistant（含 tool_calls）原样加回历史
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await _exec_tool(client, name, args, model, excluded_domains)
            if name == "web_search":
                n_search += 1
            elif name == "web_fetch" and direct_fetch_enabled:
                n_fetch += 1
            traj.append(TrajectoryStep(step, "tool_result", tool=name, args=args,
                                       result_preview=result[:300]))
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": result})

    # 撞到步数上限仍没写报告 → 强制收尾（去掉工具，要求成文）
    messages.append({
        "role": "user",
        "content": "You have reached the research limit. Using everything you found above, "
                   "write the complete final report now. Do not call any more tools.",
    })
    try:
        data = await client.chat("research", {"model": model, "messages": messages})
        report = message_text(data).strip()
        traj.append(TrajectoryStep(config.RESEARCH_MAX_STEPS, "assistant", text=report,
                                   args={"forced_finalize": True}))
        ok = bool(report)
        return ResearchResult(request_id, report, ok, traj, n_search, n_fetch,
                              config.RESEARCH_MAX_STEPS, "" if ok else "empty after finalize")
    except Exception as e:  # noqa: BLE001
        return ResearchResult(request_id, "", False, traj, n_search, n_fetch,
                              config.RESEARCH_MAX_STEPS, str(e))


async def run_research_agent(
    client: ChatClient, task: Task, model: str, excluded_domains: list[str]
) -> ResearchResult:
    """Compatibility wrapper for existing Task-based callers."""
    return await run_research_messages(client, task.problem, model, excluded_domains, task.id)
