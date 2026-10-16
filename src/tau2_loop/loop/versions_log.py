"""`loop/<domain>/versions.md`: every answering-agent version of a domain in build order.

The Optimise tab draws only the versions played on every task of their split (`full` in
`loop/history.py`). The others were each played on part of a split, to check one change on
the conversations it targeted, or never played; a pass count of theirs cannot stand beside a
full run's, but together they are how the optimisation went between the versions the tab
draws. `make versions-log DOMAIN=` writes them here, between the versions they led to, from
the same reading `GET /api/versions` serves; it calls no model and changes no record.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tau2_loop.config import LOOP_DIR


def path(domain: str) -> Path:
    return LOOP_DIR / domain / "versions.md"


def _num(name: str) -> int:
    m = re.fullmatch(r"v(\d+)", name)
    return int(m.group(1)) if m else -1


def spans(names: list[str]) -> str:
    """`v5, v6, v9–v22`: three or more consecutive version numbers as one span."""
    out: list[list[str]] = []
    for n in names:
        if out and _num(n) == _num(out[-1][-1]) + 1:
            out[-1].append(n)
        else:
            out.append([n])
    return ", ".join(f"{g[0]}–{g[-1]}" if len(g) > 2 else ", ".join(g) for g in out)


def _model(v: dict[str, Any]) -> str:
    m = re.sub(r"^claude-", "", re.sub(r"^claude-sdk/", "", v.get("model") or ""))
    return " · ".join(x for x in (m, v.get("effort")) if x) or "no run"


def _score(r: dict[str, Any] | None, split: str) -> str:
    """`train 38 of 60`; a run on part of a split says so: `train 1 of 2 played, of the split's 60`."""
    if not r:
        return f"no {split} run"
    s = f"{split} {r['passed']} of {r['n']}"
    if r.get("cut") is None:
        s += " played" + (f", of the split's {r['of']}" if r.get("of") else "")
    return s


def _cell(r: dict[str, Any] | None) -> str:
    if not r:
        return "—"
    return (
        f"{r['passed']}/{r['n']}"
        if r.get("cut") is not None
        else f"{r['passed']} of {r['n']} played"
    )


def _verdict(v: dict[str, Any]) -> str:
    moved = f" (+{v['fixed']} −{v['broke'] or 0})" if v.get("fixed") is not None else ""
    match v.get("verdict"):
        case "promote":
            return f"promoted{moved}"
        case "hold":
            return f"held{moved}"
        case "first":
            return "first champion"
        case "by hand":
            return "promoted by hand"
        case None:
            return "never played" if not v.get("train") and not v.get("test") else "not gated"
        case other:
            return str(other)


def _made(v: dict[str, Any]) -> str:
    m = v["made"]
    src = " ".join(
        x
        for x in (
            f"cycle {m['cycle']}" if m.get("cycle") is not None else "",
            f"from {m['source']}" if m.get("source") else "",
        )
        if x
    )
    detail = f": {m['detail']}" if m.get("detail") else ""
    return f"{m['kind']}{f' {src}' if src else ''}{detail}"


def _entry(v: dict[str, Any], heading: str) -> list[str]:
    runs = [r["run_id"] for r in (v.get("train"), v.get("test")) if r and r.get("run_id")]
    lines = [
        f"{heading}{' · champion' if v.get('champion') else ''}",
        "",
        f"- **Made:** {_made(v)}",
        f"- **Played:** {_score(v.get('train'), 'train')} · {_score(v.get('test'), 'test')} · {_model(v)} · {_verdict(v)}",
    ]
    if v.get("why"):
        lines.append(f"- **Why:** {v['why']}")
    if runs:
        lines.append(f"- **Runs:** {', '.join(f'`{r}`' for r in runs)}")
    return [*lines, ""]


def render(h: dict[str, Any]) -> str:
    """The log for one domain's `version_history`, as Markdown."""
    d = h["domain"]
    vs = [{**v, "champion": v["version"] == h.get("champion")} for v in h["versions"]]
    kept = [v["version"] for v in vs if v.get("full")]
    off = [v["version"] for v in vs if not v.get("full")]
    out = [
        f"# {d}: every answering-agent version, in build order",
        "",
        f"The Optimise tab draws the {len(kept)} of {len(vs)} versions played on every task of their split: "
        f"{spans(kept) or 'none'}. "
        + (
            f"The other {len(off)} ({spans(off)}) were each played on part of a split, to check one change on "
            "the conversations it targeted, or never played, so a pass count of theirs cannot stand beside a full "
            "run's. They are how the optimisation went between the versions the tab draws, so they are kept here, "
            "between the versions they led to."
            if off
            else "None was left off."
        ),
        "",
        f"Written by `make versions-log DOMAIN={d}` from `agents/{d}/v*/`, `loop/{d}/ledger.jsonl`, "
        f"`loop/{d}/registry.json` and the run folders, the reading `GET /api/versions` serves. "
        "Re-run it when a version is added or played; do not edit it by hand.",
        "",
        "| version | on the tab | made | from | train | test | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in vs:
        out.append(
            f"| {v['version']} | {'yes' if v.get('full') else 'no'} | {v['made']['kind']} | {v['made'].get('source') or '—'} "
            f"| {_cell(v.get('train'))} | {_cell(v.get('test'))} | {_verdict(v)} |"
        )
    out.append("")
    prev: str | None = None
    i = 0
    while i < len(vs):
        v = vs[i]
        if v.get("full"):
            out += _entry(v, f"## {v['version']} · on the Optimise tab")
            prev = v["version"]
            i += 1
            continue
        j = i
        while j < len(vs) and not vs[j].get("full"):
            j += 1
        group = vs[i:j]
        nxt = vs[j]["version"] if j < len(vs) else None
        where = (
            f"Between {prev} and {nxt}"
            if prev and nxt
            else f"After {prev}"
            if prev
            else f"Before {nxt}"
            if nxt
            else "Every version"
        )
        out += [
            f"## {where} · {len(group)} {'version' if len(group) == 1 else 'versions'} left off the tab",
            "",
        ]
        for g in group:
            out += _entry(g, f"### {g['version']}")
        i = j
    return "\n".join(out).rstrip() + "\n"


def write(domain: str) -> tuple[Path, dict[str, Any]]:
    """Render the domain's log from the committed files and write it; returns the path and history."""
    from tau2_loop.loop.history import load_history

    h = load_history(domain)
    p = path(domain)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(h))
    return p, h
