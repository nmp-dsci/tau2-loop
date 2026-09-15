"""The one place a model is named or a billing path chosen.

Every model call in this repo — the agent under test, tau2's user simulator,
tau2's NL-assertion judge, the optimiser — goes through the Claude Agent SDK on
the subscription. Nothing builds an Anthropic client or reads an API key. The
demo image cannot call a model at all: `require_live()` raises under DEMO_MODE
before any session starts.
"""

from __future__ import annotations

import os
from typing import Literal

from tau2_loop.config import settings

MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}

# tau2 routes a model string through litellm; this prefix routes it to the SDK
# provider instead (`sdk_provider.py`), so `claude-sdk/claude-haiku-4-5` is a
# Haiku call on the subscription.
SDK_PREFIX = "claude-sdk/"

# Every Agent SDK session in this app runs at this effort unless a version's
# `agent.yaml` pins another one for the task agent.
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT: Effort = "medium"


class BillingError(RuntimeError):
    """The environment would bill the wrong way, or cannot bill at all."""


def resolve_model(name: str) -> str:
    """`haiku` → `claude-haiku-4-5`; a full model id passes through; the SDK prefix is stripped."""
    name = name.strip()
    if name.startswith(SDK_PREFIX):
        name = name[len(SDK_PREFIX) :]
    return MODELS.get(name.lower(), name)


def sdk_model(name: str) -> str:
    """The litellm-facing model string for a tau2 role: `claude-sdk/<full id>`."""
    return SDK_PREFIX + resolve_model(name)


def short_model(model: str) -> str:
    model = resolve_model(model)
    for alias, full in MODELS.items():
        if full == model:
            return alias
    return model.replace("claude-", "")


def scrub_injected_key() -> bool:
    """Drop an ANTHROPIC_API_KEY that a dotenv search injected after start-up.

    tau2 calls `load_dotenv()` on import with an upward directory search, so a
    `~/.env` holding a key lands in `os.environ` without the user exporting it.
    A key the shell had at start-up is the user's choice and is left for
    `require_live()` to refuse; an injected one is removed so the subscription
    path stays the only path. Returns True when something was scrubbed.
    """
    from tau2_loop.config import KEY_IN_SHELL_AT_START

    if KEY_IN_SHELL_AT_START or not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return False
    os.environ.pop("ANTHROPIC_API_KEY", None)
    return True


def require_live() -> None:
    """Refuse to start a model call in a state where the bill would be a surprise."""
    s = settings()
    if s.demo_mode:
        raise BillingError(
            "DEMO_MODE=1: this deployment serves committed runs and never calls a model"
        )
    scrub_injected_key()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if s.billing == "subscription" and key:
        raise BillingError(
            "ANTHROPIC_API_KEY is set while BILLING=subscription. Unset one: with a key present the "
            "CLI bills per token even though the subscription would cover the call."
        )
    if s.billing == "api" and not key:
        raise BillingError("BILLING=api but ANTHROPIC_API_KEY is not set")


def subscription_env() -> dict[str, str]:
    """Environment for the Agent SDK child process, with per-token billing made impossible.

    Ported from DABStep-loop / ConvFinQA-agent: an `ANTHROPIC_API_KEY` in the
    child makes the CLI bill per token silently, and `CLAUDE_CODE_*` variables
    make a child started from inside a Claude Code session bill against that
    session. The SDK merges this mapping over the parent's environment, so
    omitting a key is not enough — it has to be blanked.
    """
    drop = {"ANTHROPIC_API_KEY", "CLAUDECODE"}
    env = {
        k: v for k, v in os.environ.items() if k not in drop and not k.startswith("CLAUDE_CODE_")
    }
    if settings().billing == "subscription":
        env["ANTHROPIC_API_KEY"] = ""
        env["CLAUDECODE"] = ""
    return env
