"""Agent versions are folders: `agents/<domain>/vN/{system.md, agent.yaml, helper.py}`.

Two of those files are the optimiser's only surfaces (`system.md`, `helper.py`);
`agent.yaml` is frozen across a loop cycle so a comparison is between prompts
and helpers, not between models or budgets. The fingerprint is what a run is
logged against, so two runs of the same bytes compare and two runs of
different bytes never masquerade as one version. Versions are per domain: an
airline prompt and a retail prompt evolve on their own ledgers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from tau2_loop.config import AGENTS_DIR

SURFACES = ("system.md", "helper.py")
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

    @property
    def ref(self) -> str:
        return f"{self.domain}/{self.name}"

    @property
    def helper_path(self) -> Path | None:
        p = self.path / "helper.py"
        return p if p.exists() else None

    def files(self) -> dict[str, str]:
        out = {
            "system.md": self.system_prompt,
            "agent.yaml": (self.path / "agent.yaml").read_text(),
        }
        if self.helper is not None:
            out["helper.py"] = self.helper
        return out


def _fingerprint(path: Path) -> str:
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
