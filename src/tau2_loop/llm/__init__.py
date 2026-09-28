"""The harness's view of the models: names, routes and the billing guard.

Every model call in this repo — the agent under test, tau2's user simulator,
tau2's NL-assertion judge, the optimiser — goes through the Claude Agent SDK on
the subscription. Nothing builds an Anthropic client or reads an API key. The
demo image cannot call a model at all: `require_live()` raises under DEMO_MODE
before any session starts. The model names, the effort and the billing check
live in `core.py`, the sealed part the agent service ships alone; this module
adds what only the harness needs (the litellm prefix, the dotenv scrub, the
optimiser's environment, redaction of files).
"""

from __future__ import annotations

import os
from pathlib import Path

from tau2_loop.config import settings
from tau2_loop.llm.core import (
    EFFORT,
    MODELS,
    REDACTED_EMAIL,
    BillingError,
    Effort,
    account_email,
    check_billing,
)
from tau2_loop.llm.core import resolve_model as _core_resolve

# tau2 routes a model string through litellm; this prefix routes it to the SDK
# provider instead (`sdk_provider.py`), so `claude-sdk/claude-haiku-4-5` is a
# Haiku call on the subscription.
SDK_PREFIX = "claude-sdk/"

__all__ = [
    "EFFORT",
    "MODELS",
    "REDACTED_EMAIL",
    "SDK_PREFIX",
    "BillingError",
    "Effort",
    "account_email",
]


def resolve_model(name: str) -> str:
    """`haiku` → `claude-haiku-4-5`; a full model id passes through; the route prefix is stripped."""
    return _core_resolve(name)


def sdk_model(name: str) -> str:
    """The litellm-facing model string for a tau2 role: `claude-sdk/<full id>`."""
    return SDK_PREFIX + resolve_model(name)


def model_label(model: str) -> str:
    """`claude-sonnet-5` → `Claude Sonnet 5`, `haiku` → `Claude Haiku 4.5`: the name a prompt says."""
    parts = resolve_model(model).split("-")
    if len(parts) < 2 or parts[0] != "claude":
        return resolve_model(model)
    return f"Claude {parts[1].capitalize()} {'.'.join(parts[2:])}".strip()


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
    scrub_injected_key()
    check_billing()


def subscription_env() -> dict[str, str]:
    """Environment for the optimiser's SDK session, with per-token billing made impossible.

    The task agent, the user simulator and the judge run on `core.sealed_env()`,
    an allow-list; the optimiser's session works on the repo with Bash, so it
    keeps the parent's environment minus what would bill the wrong way.

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
    # DABStep-loop s02: the CLI names every session with a separate Haiku call
    # (≈1.9k tokens). Here every model call is its own session, so that call
    # would be paid per turn; this switch stops it.
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


def redact(text: str) -> str:
    """Replace the account email wherever a model reply or a file mentions it."""
    email = account_email()
    return text.replace(email, REDACTED_EMAIL) if email else text


def redact_tree(root: Path) -> int:
    """Rewrite every text file under `root` that mentions the account email; returns the count."""
    email = account_email()
    if not email:
        return 0
    n = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            text = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if email in text:
            p.write_text(text.replace(email, REDACTED_EMAIL))
            n += 1
    return n


# The harness knobs a run records (`run.json.harness`): DABStep-loop s02 measured
# a 27k-token prefix of connector tool schemas the CLI inherits from the
# user-level claude.ai config on every API call unless the session is told to
# use only the MCP servers it is given. Nothing here touches an agent's files.
HARNESS = "lean"  # strict MCP, no session-title call
