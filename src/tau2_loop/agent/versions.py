"""Agent versions are folders: `agents/<domain>/vN/{system.md, agent.yaml, helper.py, …}`.

The optimiser's surfaces are `system.md` and `helper.py` and, since s09, three
code surfaces the harness calls around each turn: `checks.py` (write-time
checks), `memory.py` (facts kept within one conversation) and `guidance.py`
(a per-turn reminder). `agent.yaml` is frozen across a loop cycle so a
comparison is between prompts and code, not between models or budgets. The fingerprint is what a run is
logged against, so two runs of the same bytes compare and two runs of
different bytes never masquerade as one version. Versions are per domain: an
airline prompt and a retail prompt evolve on their own ledgers.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tau2_loop.config import AGENTS_DIR

# The classic optimiser's two surfaces, and the three code surfaces a routing optimiser may add (s09).
PROMPT_SURFACES = ("system.md", "helper.py")
CODE_SURFACES = ("checks.py", "memory.py", "guidance.py")
SURFACES = PROMPT_SURFACES + CODE_SURFACES
FROZEN = ("agent.yaml",)


@dataclass(frozen=True)
class AgentConfig:
    model: str = "haiku"
    effort: str = "medium"
    tool_mode: str = "json"  # json: tool calls as the JSON contract in text (the only mode in v0)
    max_steps: int = 200  # tau2's own default; the orchestrator's cap, not ours


@dataclass(frozen=True)
class AgentVersion:
    domain: str
    name: str
    path: Path
    system_prompt: str
    config: AgentConfig
    helper: str | None
    fingerprint: str
    # the code surfaces present, name → source (s09); empty for every version before them
    code: dict[str, str] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.domain}/{self.name}"

    @property
    def helper_path(self) -> Path | None:
        p = self.path / "helper.py"
        return p if p.exists() else None

    def surface_path(self, name: str) -> Path | None:
        p = self.path / name
        return p if p.exists() else None

    def files(self) -> dict[str, str]:
        out = {
            "system.md": self.system_prompt,
            "agent.yaml": (self.path / "agent.yaml").read_text(),
        }
        if self.helper is not None:
            out["helper.py"] = self.helper
        out.update(self.code)
        return out


def _fingerprint(path: Path) -> str:
    """Only the files present count, in name order: adding a surface name leaves every older
    version's fingerprint as it was."""
    h = hashlib.sha256()
    for name in sorted(SURFACES + FROZEN):
        p = path / name
        if p.exists():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def version_dir(domain: str, name: str) -> Path:
    return AGENTS_DIR / domain / name


def load_version(domain: str, name: str) -> AgentVersion:
    path = version_dir(domain, name)
    if not (path / "system.md").exists():
        raise FileNotFoundError(f"no agent at {path}")
    cfg_raw = (
        yaml.safe_load((path / "agent.yaml").read_text()) if (path / "agent.yaml").exists() else {}
    )
    config = AgentConfig(**(cfg_raw or {}))
    helper = (path / "helper.py").read_text() if (path / "helper.py").exists() else None
    return AgentVersion(
        domain=domain,
        name=name,
        path=path,
        system_prompt=(path / "system.md").read_text(),
        config=config,
        helper=helper,
        fingerprint=_fingerprint(path),
        code={n: (path / n).read_text() for n in CODE_SURFACES if (path / n).exists()},
    )


def parse_ref(ref: str, default_domain: str | None = None) -> tuple[str, str]:
    """`airline/v2` → (airline, v2); a bare `v2` needs `default_domain`."""
    if "/" in ref:
        d, n = ref.split("/", 1)
        return d, n
    if default_domain is None:
        raise ValueError(f"agent ref {ref!r} needs a domain: use <domain>/<vN>")
    return default_domain, ref


def list_versions(domain: str) -> list[AgentVersion]:
    base = AGENTS_DIR / domain
    if not base.exists():
        return []
    names = sorted(
        (p.name for p in base.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)),
        key=lambda n: int(n[1:]),
    )
    return [load_version(domain, n) for n in names]


def next_version_name(domain: str) -> str:
    versions = list_versions(domain)
    return f"v{int(versions[-1].name[1:]) + 1}" if versions else "v0"


def helper_signatures(helper_source: str) -> str:
    """The `def` lines and first docstring line of a helper, for the prompt."""
    lines: list[str] = []
    src = helper_source.splitlines()
    for i, line in enumerate(src):
        if line.startswith("def "):
            sig = line.strip()
            doc = ""
            if i + 1 < len(src) and src[i + 1].strip().startswith(('"""', "'''")):
                doc = src[i + 1].strip().strip('"').strip("'").strip()
            lines.append(f"{sig}  # {doc}" if doc else sig)
    return "\n".join(lines)


def forked_from(domain: str, name: str) -> str | None:
    """The version a hand-made fork copied its surfaces from (its diagnosis.json says), else None."""
    p = version_dir(domain, name) / "diagnosis.json"
    try:
        v = json.loads(p.read_text()).get("forked_from")
    except (OSError, ValueError):
        return None
    return str(v) if v else None


def lineage(domain: str, name: str) -> list[str]:
    """`name`, then the version it was forked from, and so on back: v3 forked from v0 → [v3, v0].
    It stops at a source whose folder is gone (banking's v0, retired when v1 became its base)."""
    out = [name]
    while (
        (parent := forked_from(domain, out[-1]))
        and parent not in out
        and version_dir(domain, parent).exists()
    ):
        out.append(parent)
    return out


def base_version(domain: str) -> str:
    """The version a plain `make eval` runs: v0, or the oldest version left where v0 was retired
    (banking: v1), never a later champion."""
    versions = list_versions(domain)
    return versions[0].name if versions else "v0"


AGENT_YAML_HEADER = (
    "# Frozen across a loop cycle: the optimiser edits the version's surfaces, never this file.\n"
)


def fork_version(
    domain: str, source: str, model: str | None = None, effort: str | None = None
) -> AgentVersion:
    """A hand-made version: `source`'s surfaces unchanged, a different `agent.yaml`.

    The optimiser may not touch `agent.yaml`, so a model or effort change is a
    fork, not a loop cycle. Its `diagnosis.json` says so (`kind: model swap`,
    `forked_from`), so the ledger, the Optimise tab and the next optimiser
    session can tell why the version exists and which held challengers it inherits.
    """
    src = load_version(domain, source)
    cfg = src.config
    new = AgentConfig(
        model=model or cfg.model,
        effort=effort or cfg.effort,
        tool_mode=cfg.tool_mode,
        max_steps=cfg.max_steps,
    )
    if new == cfg:
        raise ValueError(f"a fork of {domain}/{source} needs a different model or effort")
    name = next_version_name(domain)
    dest = version_dir(domain, name)
    dest.mkdir(parents=True)
    for surface in SURFACES:
        if (src.path / surface).exists():
            shutil.copyfile(src.path / surface, dest / surface)
    (dest / "agent.yaml").write_text(
        AGENT_YAML_HEADER + "".join(f"{k}: {v}\n" for k, v in new.__dict__.items())
    )
    changed = [
        f"{k}: {getattr(cfg, k)} → {getattr(new, k)}"
        for k in ("model", "effort")
        if getattr(cfg, k) != getattr(new, k)
    ]
    diagnosis: dict[str, Any] = {
        "kind": "model swap",
        "forked_from": source,
        "agent_yaml": changed,
        "prompt_diff_summary": f"none: {source}'s system.md, unchanged",
        "helper_diff_summary": f"none: {source}'s helper.py, unchanged"
        if src.helper is not None
        else "none: no helper",
        "diagnoses": [],
        "expected_to_fix": [],
        "risks": [
            "a prompt tuned on one model's failures can over-steer another; the version's own "
            "train run is its baseline"
        ],
        "changes": [
            {
                "file": "agent.yaml",
                "anchor": "model",
                "what": "; ".join(changed),
                "why": "a hand-made fork: the optimiser may not change agent.yaml",
                "task_ids": [],
            }
        ],
    }
    (dest / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    return load_version(domain, name)
