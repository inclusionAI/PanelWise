"""加载 DRACO 数据集并做子集选择。

数据形态（每行一个 JSON）：
  id      : UUID
  domain  : 10 个领域之一
  problem : 研究任务（带人设/交付物/范围）
  answer  : JSON 字符串，解析后是 rubric（sections[].criteria[]）

子集选择用“按领域轮转”的分层抽样：10 子集 → 每个领域各 1 题，最大化多样性。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from . import config


@dataclass
class Criterion:
    id: str
    weight: int
    requirement: str
    section: str  # factual-accuracy / breadth-and-depth-of-analysis / ...

    @property
    def negative(self) -> bool:
        return self.weight < 0


@dataclass
class Task:
    id: str
    domain: str
    problem: str
    criteria: list[Criterion]

    @property
    def positive_weight_total(self) -> int:
        return sum(c.weight for c in self.criteria if c.weight > 0)


def _local_path() -> str:
    return os.path.join("data", "test.jsonl")


def download(force: bool = False) -> str:
    """下载 DRACO test.jsonl 到 data/，返回本地路径（已存在则跳过）。"""
    path = _local_path()
    if os.path.exists(path) and not force:
        return path
    os.makedirs("data", exist_ok=True)
    headers = {}
    if config.HF_TOKEN:
        headers["Authorization"] = f"Bearer {config.HF_TOKEN}"
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(config.DRACO_JSONL_URL, headers=headers)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    return path


def _parse_rubric(answer_field: str) -> list[Criterion]:
    rubric = json.loads(answer_field) if isinstance(answer_field, str) else answer_field
    out: list[Criterion] = []
    for section in rubric.get("sections", []):
        sec_id = section.get("id", "")
        for c in section.get("criteria", []):
            out.append(
                Criterion(
                    id=str(c["id"]),
                    weight=int(c["weight"]),
                    requirement=str(c.get("requirement", "")),
                    section=sec_id,
                )
            )
    return out


def load_all() -> list[Task]:
    path = download()
    tasks: list[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            tasks.append(
                Task(
                    id=str(row["id"]),
                    domain=str(row.get("domain", "unknown")),
                    problem=str(row["problem"]),
                    criteria=_parse_rubric(row["answer"]),
                )
            )
    return tasks


def select_subset(tasks: list[Task], n: int) -> list[Task]:
    """按领域轮转的分层抽样，取 n 题（确定性，便于复现）。"""
    if n >= len(tasks):
        return list(tasks)

    by_domain: dict[str, list[Task]] = {}
    for t in sorted(tasks, key=lambda x: x.id):
        by_domain.setdefault(t.domain, []).append(t)

    domains = sorted(by_domain.keys())
    selected: list[Task] = []
    idx = 0
    while len(selected) < n:
        progressed = False
        for d in domains:
            if idx < len(by_domain[d]):
                selected.append(by_domain[d][idx])
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
        idx += 1
    return selected
