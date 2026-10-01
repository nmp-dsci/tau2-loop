"""A judge version, and what it reads at a checkpoint: the one function replay and the live hook share.

A version is a folder, `judges/<domain>/<kind>/<name>/`, holding `judge.md` (the rubric) and
`judge.yaml` (model, effort, threshold, trigger). Its fingerprint is the hash of both files, as an
agent version's is of its surfaces.

The judge's input is what the agent had, and nothing else: the rubric, the policy and the tools in
a system prompt that is the same on every call (so the prompt cache reads it back), then the
conversation up to the checkpoint, numbered, one block per message, then the reply the agent
proposes. It never sees gold, the user simulator's instructions, the grader, or another
conversation. J5's live hook calls `build` on the live history, so a replayed verdict and a live
one read the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tau2_loop.config import ROOT

JUDGES_DIR = ROOT / "judges"


@dataclass(frozen=True)
class JudgeVersion:
    domain: str
    kind: str
    name: str
    path: Path
    rubric: str
    model: str
    effort: str
    threshold: float
    structured: bool
    fingerprint: str
    config: dict[str, Any]

    @property
    def ref(self) -> str:
        return f"{self.domain}/{self.kind}/{self.name}"


def load(domain: str, name: str, kind: str = "plan") -> JudgeVersion:
    path = JUDGES_DIR / domain / kind / name
    md, yml = (path / "judge.md").read_text(), (path / "judge.yaml").read_text()
    cfg = yaml.safe_load(yml) or {}
    return JudgeVersion(
        domain=domain,
        kind=kind,
        name=name,
        path=path,
        rubric=md,
        model=str(cfg.get("model", "claude-sonnet-5")),
        effort=str(cfg.get("effort", "medium")),
        threshold=float(cfg.get("threshold", 0.5)),
        structured=bool(cfg.get("structured", True)),
        fingerprint=hashlib.sha256((md + "\0" + yml).encode()).hexdigest()[:12],
        config=cfg,
    )


def system_prompt(judge: JudgeVersion, policy: str, tools: list[dict[str, Any]]) -> str:
    tool_lines = "\n".join(
        f"- {t['name']} ({t.get('type')}): {t.get('description') or ''}" for t in tools
    )
    return f"{judge.rubric.rstrip()}\n\n## The airline policy\n{policy.strip()}\n\n## The agent's tools (name, type, description)\n{tool_lines}\n"


def _line(i: int, m: dict[str, Any], names: dict[str, str]) -> str:
    role = m.get("role")
    if role == "tool":
        return (
            f"[{i}] tool result ({names.get(str(m.get('id')), 'tool')}): {m.get('content') or ''}"
        )
    who = "agent" if role == "assistant" else "customer" if role == "user" else str(role)
    parts = [f"[{i}] {who}: {m['content']}"] if m.get("content") else []
    for tc in m.get("tool_calls") or []:
        parts.append(
            f"[{i}] {who} → {tc['name']}({json.dumps(tc.get('arguments') or {}, ensure_ascii=False)})"
        )
    return "\n".join(parts) if parts else f"[{i}] {who}: (empty)"


def blocks(msgs: list[dict[str, Any]], at: int, reply: str) -> list[str]:
    """The conversation before message `at`, one block per message, then the proposed reply.

    Blocks, not one string, so a later checkpoint in the same conversation reads every earlier
    message back from the prompt cache (s10's one-mark-on-the-last-block route)."""
    names = {str(tc.get("id")): tc["name"] for m in msgs[:at] for tc in (m.get("tool_calls") or [])}
    out = ["## The conversation so far"] + [_line(i, m, names) for i, m in enumerate(msgs[:at])]
    out.append(
        f"## The agent's proposed reply, not yet shown to the customer\n[{at}] agent: {reply}\n\n"
        "Review it and answer in the schema."
    )
    return out
