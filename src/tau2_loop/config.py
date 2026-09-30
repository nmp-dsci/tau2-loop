"""Paths and settings. One place; nothing else reads `os.environ` for these.

`Settings` boots keyless: the absence of a key is a legitimate state (demo mode,
CI, the gate) and only a call that would actually reach a model asks for one.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

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
# The cut per domain (data/splits.py). Version 1 drew 20 train + 20 test and
# held the rest in reserve; version 2 keeps those 40 where they were and deals
# the reserve out evenly, so train and test are each half of the base set.
SPLIT_VERSION = 2
# Banking's retrieval (s09 Q1): BM25 plus tau2's grep over the knowledge base, both local.
# The board's AllTools adds OpenAI embeddings, which the subscription cannot call.
BANKING_RETRIEVAL = "bm25_grep"
# Domains whose train split is dealt into a read half (the optimiser's) and a gate half
# (the gate's, never shown to an optimiser): s09 option B.
HALVED_DOMAINS: tuple[str, ...] = ("banking_knowledge",)
# A test split capped to its first n tasks (v1's test first, then the dealt ones), the rest held
# back in reserve, never moved to train: banking's 49 cost ~50 minutes a version (s09).
TEST_CAP: dict[str, int] = {"banking_knowledge": 25}
V1_SIZE = 20

# tau2 reads its data dir from this variable; the submodule's own `data/` is the
# pinned copy, so point there before anything imports tau2.
os.environ.setdefault("TAU2_DATA_DIR", str(TAU2_DATA_DIR))

# What the shell had before anything ran. `tau2.utils.utils` calls
# `load_dotenv()` with a directory search on import, which finds a `~/.env`
# and injects whatever keys live there; `llm.require_live()` uses this to tell
# an injected ANTHROPIC_API_KEY (scrubbed) from one the user exported (refused).
KEY_IN_SHELL_AT_START = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


class Settings(BaseModel):
    """Runtime settings, read once from the environment.

    The two central services are nmp-central-ai's (PLATFORM.md): MLflow on :5000 and
    Postgres on :5432, database `tau2`. Both are optional at boot — the viewer serves
    committed files with either one stopped — but neither is ever replaced by a local
    store (PLATFORM.md rule 1). `make -C ../nmp-central-ai db-urls` prints the URLs.
    """

    billing: str = "subscription"
    demo_mode: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    database_url: str = "postgresql://tau2_owner:tau2_owner@localhost:5432/tau2"
    ro_database_url: str = "postgresql://tau2_ro:tau2_ro@localhost:5432/tau2"
    pg_superuser_url: str = "postgresql://nmp:nmp@localhost:5432/tau2"
    code_sha: str = "unknown"
    # The agent service (llm/service.py). With a URL set, the task agent's calls go over
    # HTTP to it; the user simulator and the judge stay in-process on the same core.
    agent_service_url: str = ""
    agent_service_token: SecretStr = SecretStr("")

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        default = cls.model_fields
        return cls(
            billing=env("BILLING", "subscription").strip().lower(),
            demo_mode=env("DEMO_MODE", "").strip() in {"1", "true", "yes"},
            mlflow_tracking_uri=env("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            database_url=env("DATABASE_URL", str(default["database_url"].default)),
            ro_database_url=env("RO_DATABASE_URL", str(default["ro_database_url"].default)),
            pg_superuser_url=env("PG_SUPERUSER_URL", str(default["pg_superuser_url"].default)),
            code_sha=env("TAU2LOOP_CODE_SHA", "unknown"),
            agent_service_url=env("AGENT_SERVICE_URL", "").strip().rstrip("/"),
            agent_service_token=SecretStr(env("AGENT_SERVICE_TOKEN", "").strip()),
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
