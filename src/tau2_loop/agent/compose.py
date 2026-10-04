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
HELPER_HOOKS = ("extra_context", "on_tool_call", "on_reply")
# s09: the code surfaces and the one hook each defines, in the order a turn meets them.
#   memory.py    remember(state, name, arguments, result) -> None   after each tool result and user message
#   guidance.py  guidance(state, trigger) -> str | None              before each model call ("user" | "tool")
#   checks.py    check_write(name, arguments, state) -> str | None   on each write call; a string blocks it once
CODE_HOOKS = {"memory.py": "remember", "guidance.py": "guidance", "checks.py": "check_write"}
#   checks.py    check_reply(text, state) -> str | None   on each text reply; a string sends it back once
#                (s14: a rule about what the agent says, e.g. refusing a fourth transfer request)
REPLY_HOOK = ("checks.py", "check_reply")
HOOKS = HELPER_HOOKS + tuple(CODE_HOOKS.values()) + (REPLY_HOOK[1],)
# A per-turn reminder is cut to this many characters (s09 §5, the guidance budget).
GUIDANCE_CHARS = 600
# The Claude CLI tells every session the real date in a system reminder, and nothing switches it
# off but `--bare`, which drops the subscription login. The agent may know only the time its world
# gives it (the policy's "The current time is …", or a tool), so its prompt ends with this. A run
# records it in `RunMeta.sim_rules`; one without ran before it, with no note.
CLOCK_NOTE = (
    "A system reminder may tell you today's date. It is not this conversation's date: the current "
    "time is only what the policy above or a tool says."
)

# The CLI also tells every session its account's email, and the agent takes it for the
# customer's: v1 sent it in 1,192 tool calls across 30 of banking's 97 conversations (s13). A
# version whose `agent.yaml` says `identity_note: true` gets this after the clock note. It is a
# frozen setting, like the model, so a version without it composes exactly as before.
IDENTITY_NOTE = (
    "The session may name an account email. It is not the customer's and belongs to no one in "
    "this conversation: identify the customer only from what they tell you or a tool returns, "
    "and never use that email in a tool call."
)


@dataclass(frozen=True)
class Composed:
    """The prompt and its parts: `text` is exactly what is sent; the rest say where it came from."""

    text: str
    system_md: str
    policy: str
    extra_context: str | None
    slotted: bool
    clock_note: str | None = None
    identity_note: str | None = None


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


def load_code_surfaces(folder: Path, prefix: str) -> dict[str, types.ModuleType]:
    """Each code surface present in a version (or run snapshot) folder, imported; a file that
    does not import is left out, as a broken helper is."""
    out: dict[str, types.ModuleType] = {}
    for name in CODE_HOOKS:
        mod = load_helper_file(folder / name, f"{prefix}_{name.removesuffix('.py')}")
        if mod is not None:
            out[name] = mod
    return out


def hooks_defined(
    helper: types.ModuleType | None, code: dict[str, types.ModuleType] | None = None
) -> dict[str, bool]:
    out = {h: callable(getattr(helper, h, None)) if helper else False for h in HELPER_HOOKS}
    for name, hook in [*CODE_HOOKS.items(), REPLY_HOOK]:
        mod = (code or {}).get(name)
        out[hook] = callable(getattr(mod, hook, None)) if mod else False
    return out


def compose(
    system_md: str,
    policy: str,
    helper: types.ModuleType | None,
    clock: bool = True,
    identity: bool = False,
) -> Composed:
    """`system.md` with the policy in its slot (or appended), then `extra_context(policy)`, then
    `CLOCK_NOTE` (`clock=False` for a run made before it), then `IDENTITY_NOTE` for a version
    whose `agent.yaml` asks for it."""
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
    if clock:
        text = f"{text}\n\n{CLOCK_NOTE}"
    if identity:
        text = f"{text}\n\n{IDENTITY_NOTE}"
    return Composed(
        text=text,
        system_md=system_md,
        policy=policy,
        extra_context=extra,
        slotted=slotted,
        clock_note=CLOCK_NOTE if clock else None,
        identity_note=IDENTITY_NOTE if identity else None,
    )
