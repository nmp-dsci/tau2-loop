"""A merge written as edits (r4, `merge_write: edits`): what changes on the library's newest version
of one job, not the whole job again.

r3 rewrote every job it merged in full, and its merges wrote a median 46k output tokens a question,
61% of them the workflows copied back (8 Oct 2026). An edit names the list and the item it touches
(steps by `id`, tools by `tool`, info by `field`, rules by `rule`; an alias is itself), so the reply
carries only the change. The harness applies the edits to the version the merge read, and the job
that comes out is checked by the same rubric as a job written in full: the edits are a shorter way
to write it, not a looser check.

Each edit is also a changelog line: an `add` is "added", a `change` or `set` is "changed", a
`remove` is "removed", with the edit's `why`, `quote` and `doc`. A removal names the item it removes,
its quotes included, so rubric D8 counts it as named; a change does not carry the old item's quotes,
so a change that drops a quote must still say so in its `why`.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

KEYS = {"steps": "id", "tools": "tool", "info": "field", "rules": "rule"}
LISTS = (*KEYS, "aliases")
FIXED = ("job", "into", "aliases", "changelog", "edits", *KEYS)  # never `set`
CHANGE = {"add": "added", "change": "changed", "set": "changed", "remove": "removed"}


class EditError(ValueError):
    """An edit that cannot be applied; the message says which and why, for the send-back."""


def is_edits(job: Any) -> bool:
    return isinstance(job, dict) and isinstance(job.get("edits"), list)


def base_name(job: dict[str, Any]) -> str | None:
    """The one library job the edits apply to: the only name in `into`."""
    into = job.get("into") or []
    into = [into] if isinstance(into, str) else into
    names = [n for n in into if isinstance(n, str) and n.strip()]
    return names[0].strip() if len(names) == 1 else None


def _squash(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _key(lst: str, item: Any) -> str:
    if lst == "aliases":
        return str(item)
    return str(item.get(KEYS[lst], "")) if isinstance(item, dict) else ""


def _find(items: list[Any], lst: str, key: Any, n: int) -> int:
    """The index of the item `key` names: its key exactly, else the one item whose key starts
    with it (a rule is named by its text, and a long one by its start)."""
    if not isinstance(key, str) or not key.strip():
        raise EditError(f"edit {n}: `key` names no item of {lst}")
    want = _squash(key)
    exact = [i for i, it in enumerate(items) if _squash(_key(lst, it)) == want]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise EditError(f"edit {n}: {len(exact)} items of {lst} are {key!r}; write the job in full")
    start = [i for i, it in enumerate(items) if _squash(_key(lst, it)).startswith(want)]
    if len(start) == 1 and len(want) >= 12:
        return start[0]
    have = ", ".join(repr(_key(lst, it)[:40]) for it in items[:12])
    raise EditError(
        f"edit {n}: no single item of {lst} is {key!r}"
        + (" (more than one starts so)" if len(start) > 1 else "")
        + f"; {lst} holds {have}{' …' if len(items) > 12 else ''}"
    )


def _quoted(item: Any) -> str:
    """An item's quotes, for a removal's changelog line, so D8 counts them as named."""
    from tau2_loop.workflows.rag_agent import quotes

    return " | ".join(q["quote"] for q in quotes(item))


def apply(entry: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """The job `entry`'s edits make of `base` (the library's newest version of the job it names in
    `into`): every field of `base`, the edits applied in order, and the entry's own `job`, `into`
    and `aliases` (added to the base's). Its changelog is the edits', then any the entry wrote."""
    job: dict[str, Any] = copy.deepcopy(base)
    for k in ("into", "changelog"):
        job.pop(k, None)
    log: list[dict[str, Any]] = []
    for n, e in enumerate(entry.get("edits") or [], 1):
        if not isinstance(e, dict):
            raise EditError(f"edit {n} is not an object")
        op = e.get("op")
        if op not in CHANGE:
            raise EditError(f"edit {n}: op {op!r} is not add, change, remove or set")
        if op == "set":
            field = e.get("field")
            if not isinstance(field, str) or field in FIXED or "value" not in e:
                raise EditError(
                    f"edit {n}: set takes a `field` (not {', '.join(FIXED)}) and its `value`"
                )
            job[field] = e["value"]
            what = field
        else:
            lst = e.get("list")
            if lst not in LISTS:
                raise EditError(f"edit {n}: `list` must be one of {', '.join(LISTS)}")
            items = job.setdefault(lst, [])
            if not isinstance(items, list):
                raise EditError(f"edit {n}: the base's {lst} is not a list")
            if op == "add":
                item = e.get("item")
                if lst == "aliases":
                    if not isinstance(item, str) or not item.strip():
                        raise EditError(f"edit {n}: an alias is a name")
                    if item not in items:
                        items.append(item)
                    what = item
                else:
                    if not isinstance(item, dict) or not _key(lst, item):
                        raise EditError(f"edit {n}: the item added to {lst} needs its {KEYS[lst]}")
                    if any(_squash(_key(lst, it)) == _squash(_key(lst, item)) for it in items):
                        raise EditError(
                            f"edit {n}: {lst} already has {_key(lst, item)!r}; change it instead"
                        )
                    at = len(items)
                    if e.get("after") is not None:
                        at = _find(items, lst, e["after"], n) + 1
                    elif e.get("before") is not None:
                        at = _find(items, lst, e["before"], n)
                    items.insert(at, item)
                    what = f"{lst} {_key(lst, item)}"
            else:
                i = _find(items, lst, e.get("key"), n)
                old = items[i]
                what = f"{lst} {_key(lst, old)}"
                if op == "remove":
                    items.pop(i)
                    if lst != "aliases" and _quoted(old):
                        what = f"{what} (quoting: {_quoted(old)})"
                elif lst == "aliases":
                    raise EditError(f"edit {n}: an alias is added or removed, not changed")
                elif isinstance(e.get("item"), dict):
                    items[i] = e["item"]
                elif isinstance(e.get("fields"), dict) and isinstance(old, dict):
                    items[i] = {**old, **e["fields"]}
                else:
                    raise EditError(
                        f"edit {n}: a change gives the whole `item`, or the `fields` it sets"
                    )
                if op == "change" and not _key(lst, items[i]):
                    raise EditError(f"edit {n}: the changed item lost its {KEYS[lst]}")
        log.append(
            {
                "change": CHANGE[op],
                "what": what,
                **{k: e[k] for k in ("why", "quote", "doc") if e.get(k)},
            }
        )
    job["job"] = str(entry.get("job") or base.get("job"))
    job["into"] = list(entry.get("into") or [])
    extra = entry.get("aliases") or []
    aliases = [*(job.get("aliases") or []), *([extra] if isinstance(extra, str) else extra)]
    job["aliases"] = list(dict.fromkeys(a for a in aliases if isinstance(a, str) and a))
    job["changelog"] = [*log, *(entry.get("changelog") or [])]
    return job


def size(o: Any) -> int:
    """Characters of JSON, for the record of what the edits saved."""
    return len(json.dumps(o, ensure_ascii=False))
