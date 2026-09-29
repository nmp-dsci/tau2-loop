"""How a version's files become the system prompt the agent is sent, with no tau2 in sight.

`LoopAgent.system_prompt()` and the viewer's `GET /api/runs/{id}/agent` both call
`compose()`, so the prompt the Agent tab shows is the prompt the agent received,
not a second rendering of it. Kept free of tau2 imports because the demo image
serves that route and ships no tau2.
"""

from __future__ import annotations

import importlib.util
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A version folder whose system.md has no slot gets the policy appended instead.
POLICY_SLOT = "{policy}"

# The three optional hooks a helper may define, in the order a turn meets them.
HOOKS = ("extra_context", "on_tool_call", "on_reply")


@dataclass(frozen=True)
class Composed:
    """The prompt and its parts: `text` is exactly what is sent; the rest say where it came from."""

    text: str
    system_md: str
    policy: str
    extra_context: str | None
    slotted: bool


def load_helper_file(path: Path | None, module_name: str) -> types.ModuleType | None:
    """Import a `helper.py` as a module, if present; a helper that does not import is None."""
    if path is None or not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 - a helper that does not import is a version without one
        return None
    return mod


def call_hook(helper: types.ModuleType | None, name: str, *args: Any) -> Any:
    """One hook, if the helper defines it; a hook that raises returns None and never breaks a turn."""
    fn = getattr(helper, name, None) if helper is not None else None
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - a helper exception never breaks a turn
        return None


def hooks_defined(helper: types.ModuleType | None) -> dict[str, bool]:
    return (
        {h: callable(getattr(helper, h, None)) for h in HOOKS}
        if helper
        else dict.fromkeys(HOOKS, False)
    )


def compose(system_md: str, policy: str, helper: types.ModuleType | None) -> Composed:
    """`system.md` with the policy in its slot (or appended), then `extra_context(policy)`."""
    slotted = POLICY_SLOT in system_md
    text = (
        system_md.replace(POLICY_SLOT, policy)
        if slotted
        else f"{system_md}\n\n# Domain policy\n{policy}"
    )
    extra = call_hook(helper, "extra_context", policy)
    extra = extra.strip() if isinstance(extra, str) and extra.strip() else None
    if extra:
        text = f"{text}\n\n{extra}"
    return Composed(
        text=text, system_md=system_md, policy=policy, extra_context=extra, slotted=slotted
    )
