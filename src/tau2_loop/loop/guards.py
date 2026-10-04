"""What the optimiser may read, and what a version it writes may contain (s09).

Four guards, all deterministic:

- **the fence**: an optimiser session reads the policy and tools from a copy in
  its version folder (`.context/`), the traces its prompt cites, and `agents/`.
  `runs/`, `data/`, `loop/` and `vendor/` are closed to it: they hold the test
  split's and the gate half's expected actions and results, which it must never
  see. A partner challenger's folder in an A/B cycle is closed too.
- **imports**: a code surface (`checks.py`, `memory.py`, `guidance.py`) runs inside
  every conversation, so it may import only a small standard-library allow-list and
  may not open files, run code or reach a network.
- **leaks**: no file of a version may carry a customer id, name or email from any
  task's data; the test split has other customers, so such a line can only be an
  answer copied from a train task.
- **budget**: a routing optimiser's `system.md` may grow by at most
  `PROMPT_GROWTH` characters over its champion's.

Beside them, a dataset's own (s13 §5): `optimisers/<domain>/guards.py`, when it
exists, checks what that dataset's optimiser wrote (`profile_violations`).
"""

from __future__ import annotations

import ast
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

from tau2_loop.config import ROOT
from tau2_loop.data.splits import read_task_extract

# `.lavish/`: the review pages, which can quote test results (s14 P8c)
FENCED = ("runs", "data", "loop", "vendor", ".lavish")
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "functools",
        "itertools",
        "json",
        "math",
        "re",
        "statistics",
        "string",
        "typing",
    }
)
BANNED_CALLS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"})
PROMPT_GROWTH = 1500
NAME_KEYS = ("first_name", "last_name", "full_name", "name")
# banking's discoverable tools are named in arguments (`open_bank_account_4821`): a tool the
# knowledge base documents for every customer, train and test, not a customer's value
TOOL_NAME_KEYS = ("agent_tool_name", "discoverable_tool_name")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def _inside(p: Path, base: Path) -> bool:
    return p == base or base in p.parents


def fence_reason(path: str, allowed: set[Path], hidden: list[Path]) -> str | None:
    """Why a Read, Glob or Grep of `path` is refused, or None when it may go ahead."""
    if not path:
        return "give a path: a search with none covers the whole repository, fenced parts included"
    p = Path(path)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    for h in hidden:
        if _inside(p, h.resolve()):
            return "that folder is another challenger of this cycle; it is scored against yours"
    if p in allowed:
        return None
    closed = [(ROOT / f).resolve() for f in FENCED] + [h.resolve() for h in hidden]
    if any(_inside(c, p) for c in closed):
        return (
            "that path holds folders closed to the optimiser: name a folder inside your "
            "version or the traces your prompt names"
        )
    for f in FENCED:
        if _inside(p, (ROOT / f).resolve()):
            return (
                f"{f}/ is closed to the optimiser: it holds tasks and results you must not see. "
                "The policy and tools are in your version folder's .context/, and the traces "
                "you may read are the ones your prompt names."
            )
    return None


def bash_fence_reason(cmd: str, hidden: list[Path]) -> str | None:
    """Why a shell command is refused: it names a fenced folder or a hidden one."""
    for h in hidden:
        if str(h.relative_to(ROOT)) in cmd or str(h) in cmd:
            return "that folder is another challenger of this cycle"
    for f in FENCED:
        if re.search(rf"(^|[\s'\"=(/.]){re.escape(f)}/", cmd):
            return (
                f"{f}/ is closed to the optimiser; read .context/ and the traces your prompt names"
            )
    return None


def import_violations(source: str) -> list[str]:
    """What a code surface does that it may not: imports outside the allow-list, file and code calls."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"does not parse: {e.msg} (line {e.lineno})"]
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            mods = []
        out += [f"imports {m}" for m in mods if m not in ALLOWED_IMPORTS]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BANNED_CALLS
        ):
            out.append(f"calls {node.func.id}()")
    return out


def _strings(value: Any, key: str = "") -> list[tuple[str, str]]:
    """(key, string) for every string in nested arguments, a list's items under the list's key. A
    string holding a JSON object or list is read as one: banking passes a discoverable tool's
    arguments as JSON text, and the customer ids are inside it."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("{", "["):
            try:
                return _strings(json.loads(text), key)
            except ValueError:
                pass
        return [(key, value)]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _strings(v, str(k))]
    if isinstance(value, list):
        return [p for v in value for p in _strings(v, key)]
    return []


def leak_values(domain: str) -> set[str]:
    """Customer data in the domain's tasks: ids and emails in the expected actions' arguments
    and the scenarios, and the names passed as arguments. Values the policy or the tool list
    also contain are generic, not a customer's, and are left out."""
    ext = read_task_extract(domain)
    generic = str(ext.get("policy") or "") + json.dumps(ext.get("tools") or [])
    found: set[str] = set()
    for t in ext.get("tasks") or []:
        for a in (t.get("evaluation_criteria") or {}).get("actions") or []:
            for k, v in _strings(a.get("arguments") or {}):
                if k in TOOL_NAME_KEYS:
                    continue
                v = v.strip()
                is_id = len(v) >= 5 and re.search(r"[A-Za-z]", v) and re.search(r"\d", v)
                is_name = k in NAME_KEYS and len(v) >= 3
                if is_id or "@" in v or is_name:
                    found.add(v)
        scenario = json.dumps((t.get("user_scenario") or {}).get("instructions") or "")
        found |= set(_EMAIL.findall(scenario))
    return {v for v in found if v not in generic}


def leaks(files: dict[str, str], values: set[str]) -> list[str]:
    """`file: value` for each customer value a version's file contains, as a whole word."""
    out: list[str] = []
    for name, text in files.items():
        for v in sorted(values):
            if re.search(rf"(?<![\w@.-]){re.escape(v)}(?![\w@-])", text):
                out.append(f"{name}: {v}")
    return out


def profile_violations(
    source: str | None,
    files: dict[str, str],
    champion: dict[str, str],
    where: str = "guards.py",
) -> list[str]:
    """A dataset's own guards (`optimisers/<domain>/guards.py`, s13 §5), run beside the shared
    ones on what its optimiser wrote. `source` is the file as its profile read it when the cycle
    began, so a session cannot weaken the check it is about to face (None: no file, nothing to
    check). It defines `check(files, champion)`: both map a file name to its text (the new
    version's files and the champion's, as `AgentVersion.files()` gives them), and it returns one
    sentence per violation, or nothing. A guard that will not compile, has no `check`, or raises
    is a violation itself: a guard that cannot run vouches for nothing."""
    if source is None:
        return []
    name = "tau2_loop_profile_guards"
    module = types.ModuleType(name)
    module.__file__ = where
    sys.modules[name] = module  # a dataclass in the guard looks its module up by name
    try:
        exec(compile(source, where, "exec"), module.__dict__)
        check = getattr(module, "check", None)
        if not callable(check):
            return [f"{where} defines no check(files, champion)"]
        found = check(dict(files), dict(champion))
    except Exception as e:  # noqa: BLE001 - a dataset's guard never crashes the cycle; it refuses it
        return [f"{where} could not run: {type(e).__name__}: {e}"]
    finally:
        sys.modules.pop(name, None)
    if not found:
        return []
    if isinstance(found, str):
        return [found]
    return [str(x) for x in found]
