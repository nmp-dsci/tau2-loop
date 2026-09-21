"""Paths and settings. One place; nothing else reads `os.environ` for these.

`Settings` boots keyless: the absence of a key is a legitimate state (demo mode,
CI, the gate) and only a call that would actually reach a model asks for one.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TAU2_ROOT = ROOT / "vendor" / "tau2-bench"
TAU2_DATA_DIR = TAU2_ROOT / "data"
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
TASKS_DIR = DATA_DIR / "tasks"
AGENTS_DIR = ROOT / "agents"
RUNS_DIR = ROOT / "runs"
LOOP_DIR = ROOT / "loop"
WORKSPACE_DIR = ROOT / "workspace"
FRONTEND_DIST = ROOT / "frontend" / "dist"

# The four scored domains, in the order every table shows them. `mock` is the
# adapter's smoke target and never appears in a results table.
DOMAINS: tuple[str, ...] = ("airline", "retail", "telecom", "banking_knowledge")
SMOKE_DOMAIN = "mock"
SPLIT_SEED = 300
SPLIT_SIZE = 20  # per split, per domain: 20 train + 20 test

# tau2 reads its data dir from this variable; the submodule's own `data/` is the
# pinned copy, so point there before anything imports tau2.
os.environ.setdefault("TAU2_DATA_DIR", str(TAU2_DATA_DIR))

# What the shell had before anything ran. `tau2.utils.utils` calls
# `load_dotenv()` with a directory search on import, which finds a `~/.env`
# and injects whatever keys live there; `llm.require_live()` uses this to tell
# an injected ANTHROPIC_API_KEY (scrubbed) from one the user exported (refused).
KEY_IN_SHELL_AT_START = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


class Settings(BaseModel):
    """Runtime settings, read once from the environment."""

    billing: str = "subscription"
    demo_mode: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    code_sha: str = "unknown"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            billing=os.environ.get("BILLING", "subscription").strip().lower(),
            demo_mode=os.environ.get("DEMO_MODE", "").strip() in {"1", "true", "yes"},
            mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            code_sha=os.environ.get("TAU2LOOP_CODE_SHA", "unknown"),
        )


def settings() -> Settings:
    return Settings.from_env()


def ledger_path(domain: str) -> Path:
    return LOOP_DIR / domain / "ledger.jsonl"


def registry_path(domain: str) -> Path:
    return LOOP_DIR / domain / "registry.json"


def quiet_tau2() -> None:
    """tau2 logs at DEBUG through loguru by default; keep only warnings on our console."""
    try:
        from loguru import logger

        logger.remove()
        logger.add(lambda m: print(m, end=""), level="WARNING")
    except Exception:  # noqa: BLE001
        return
