"""Every version of a domain in build order: how it was made, what the gate saw, who held the title.

The Runs and Optimise tabs draw this (`/api/versions`). It joins what the repo
already records and computes nothing the gate did not:

- `agents/<domain>/vN/` — the version's `agent.yaml` and, for a hand-made fork,
  `diagnosis.json` (`kind`, `forked_from`, `agent_yaml`);
- `loop/<domain>/ledger.jsonl` — one entry per cycle: the champion, the
  challenger, the train run the gate scored and the test run reported beside it;
- `loop/<domain>/registry.json` — every `promote`, in order, which is the title's history;
- the run folders' `run.json` — each run's pass count, its cut (split v1 or v2) and model.

A version's train number is the run the gate read: the ledger's challenger run,
else the version's latest promotion, else its latest registry entry, else its
newest finished train run. Its test number is the ledger's test run, else its
newest finished test run on the same cut.

A version is `full` when every run it has covers a whole side of a split (the run's tasks are
one cut's train or test, so it has a cut) and it has a train run. One played on part of a
split (`make eval TASKS=`, to check one change on the conversations it targets) or never
played is not: the Optimise tab leaves it off, and `loop/<domain>/versions.md`
(`make versions-log`) keeps it, in build order between the versions it led to.
"""

from __future__ import annotations

import re
from typing import Any

# `make fork`'s note on every hand-made change, inside brackets or as its own sentence
FORK_NOTE = re.compile(r"\s*\(?a hand-made fork: the optimiser may not change agent\.yaml\)?\.?\s*")

# a surface's name under a version's column in the figures
SHORT = {
    "system.md": "prompt",
    "helper.py": "helper",
    "checks.py": "checks",
    "memory.py": "memory",
    "guidance.py": "guidance",
}


def _run(r: dict[str, Any] | None) -> dict[str, Any] | None:
    if r is None:
        return None
    return {
        "run_id": r["run_id"],
        "passed": r["passed"],
        "n": r["n"],
        "cut": r.get("cut"),
        # the tasks its split side held: `n` of `of` for a run on part of it
        "of": r.get("of"),
        "model": r.get("model"),
        "effort": r.get("effort"),
    }


def _whole(r: dict[str, Any] | None) -> bool:
    """A run of a whole split side: its tasks are one cut's train or test."""
    return r is not None and r.get("cut") is not None


def _why(e: dict[str, Any] | None, diag: dict[str, Any]) -> str | None:
    """Why a version was made, in its maker's words: an optimiser's summary of its prompt diff,
    or for a hand-made fork the reasons given for its changes (`make fork WHY=`)."""
    summary = (e or {}).get("prompt_diff_summary") or diag.get("prompt_diff_summary")
    if diag.get("diagnoses"):
        return str(summary) if summary else None
    # every fork's change record says it is one; that is why it is a fork, not why it was made
    whys = [FORK_NOTE.sub(" ", str(c.get("why") or "")).strip() for c in diag.get("changes") or []]
    return "; ".join(dict.fromkeys(w for w in whys if w)) or None


def _newest(
    runs: list[dict[str, Any]], agent: str, split: str, cut: int | None = None
) -> dict[str, Any] | None:
    """A version's newest run of a split side, a whole one (one cut's tasks) before any part: a
    later replay of some of its tasks (s23 replayed v23 on its 34 failures) never stands for it."""
    xs = [
        r
        for r in runs
        if r["agent"] == agent and r["split"] == split and (cut is None or r.get("cut") == cut)
    ]
    whole = [r for r in xs if _whole(r)]
    return (whole or xs)[-1] if xs else None


def version_history(
    domain: str,
    versions: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """`versions`: `{name, model, effort, diagnosis}` in build order. `runs`: finished, scored,
    non-dry runs of this domain in start order, each `{run_id, agent, split, cut, passed, n, model,
    effort}`. `ledger` and `registry` are the domain's files as read."""
    by_id = {r["run_id"]: r for r in runs}
    history = registry.get("history") or []
    promotes = [h for h in history if h.get("event") == "promote"]
    cycle_of = {e["challenger"]: e for e in ledger if e.get("challenger")}

    def scored(run_id: str | None, fallback: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """A run by id as `_run` shapes it; a registry entry's own numbers when its folder is gone."""
        if run_id and run_id in by_id:
            return _run(by_id[run_id])
        if fallback and fallback.get("passed") is not None and fallback.get("n_scored"):
            return {
                "run_id": run_id,
                "passed": fallback["passed"],
                "n": fallback["n_scored"],
                "cut": None,
                "of": None,
                "model": fallback.get("model"),
                "effort": None,
            }
        return None

    reigns = []
    for i, h in enumerate(promotes):
        s = scored(h.get("run_id"), h)
        reigns.append(
            {
                "version": h["agent"],
                "run_id": h.get("run_id"),
                "passed": s["passed"] if s else None,
                "n": s["n"] if s else None,
                "cut": s["cut"] if s else None,
                # the first promotion crowns the baseline; one with no kind was the gate's
                "kind": "first" if i == 0 else h.get("kind") or "gate",
                "at": h.get("at"),
            }
        )

    out = []
    for v in versions:
        name = v["name"]
        e = cycle_of.get(name)
        o = (e or {}).get("outcome") or {}
        diag = v.get("diagnosis") or {}
        mine = [h for h in history if h.get("agent") == name]
        promoted = [h for h in mine if h.get("event") == "promote"]

        # train: the run the gate read
        if o.get("challenger_run"):
            train = scored(o["challenger_run"])
        elif promoted:
            train = scored(promoted[-1].get("run_id"), promoted[-1])
        elif mine:
            train = scored(mine[-1].get("run_id"), mine[-1])
        else:
            train = _run(_newest(runs, name, "train"))
        test = (
            scored(o["test_run"])
            if o.get("test_run")
            else _run(_newest(runs, name, "test", train["cut"] if train else None))
        )

        # how it was made
        if e and e.get("kind"):
            made = {
                "kind": e["kind"],
                "cycle": e["cycle"],
                "source": e.get("forked_from") or e.get("champion"),
                "detail": "; ".join(e.get("agent_yaml") or []),
            }
        elif e:
            opt = e.get("optimiser_model")
            detail = f"{opt} optimiser" if opt else "optimiser"
            if e.get("optimiser_mode") == "routing":  # s09: the surfaces its diagnosis chose
                short = [SHORT.get(n, n) for n in e.get("surfaces_changed") or []]
                detail = f"routing: {'+'.join(short) or 'none'}"
            made = {
                "kind": "optimise",
                "cycle": e["cycle"],
                "source": e.get("champion"),
                "detail": detail,
            }
        elif diag.get("kind"):
            made = {
                "kind": diag["kind"],
                "cycle": None,
                "source": diag.get("forked_from"),
                "detail": "; ".join(diag.get("agent_yaml") or []),
            }
        elif diag:
            made = {
                "kind": "optimise",
                "cycle": None,
                "source": diag.get("optimised_from"),
                "detail": "no ledger entry",
            }
        else:
            made = {
                "kind": "base",
                "cycle": None,
                "source": None,
                "detail": "tau2's instruction" if name == "v0" else "hand-built",
            }

        # what decided its standing
        vs = None
        if e:
            verdict = o.get("verdict") or "pending"
            champ_train = scored(e.get("champion_run"))
            tc = o.get("test_compare") or {}
            vs = {
                "version": e.get("champion"),
                "train": champ_train,
                "test": scored(tc.get("champion_run")) if tc.get("champion_run") else None,
            }
        elif promoted:
            first = (
                bool(reigns)
                and reigns[0]["version"] == name
                and reigns[0]["run_id"] == promoted[0].get("run_id")
            )
            verdict = "first" if first else "by hand"
        else:
            # scored but never put to the gate, or entered in the registry without a promotion
            verdict = None if train is None else "registered" if mine else "not gated"

        out.append(
            {
                "version": name,
                "model": (train or {}).get("model") or v.get("model"),
                # only what the run recorded: runs before split v2 did not record effort
                "effort": (train or {}).get("effort") if train else v.get("effort"),
                "made": made,
                "train": train,
                "test": test,
                "verdict": verdict,
                "full": train is not None and _whole(train) and (test is None or _whole(test)),
                "why": _why(e, diag),
                "fixed": len(o["fixed"]) if "fixed" in o else None,
                "broke": len(o["broken"]) if "broken" in o else None,
                "p": o.get("p_value"),
                "vs": vs,
                "held_title": any(r["version"] == name for r in reigns),
            }
        )

    champ = registry.get("champion") or {}
    return {"domain": domain, "champion": champ.get("agent"), "versions": out, "reigns": reigns}


def load_history(domain: str) -> dict[str, Any]:
    """`version_history` over the committed files: every finished, scored, non-dry run of the
    domain, its version folders, ledger and registry. `GET /api/versions` serves this."""
    import json

    from tau2_loop.agent.versions import list_versions
    from tau2_loop.data.splits import cut_of, cut_size
    from tau2_loop.eval.runner import list_runs
    from tau2_loop.loop.ledger import read_ledger
    from tau2_loop.tracking.registry import read_registry

    runs = []
    for m in list_runs(domain):
        if m.dry_run or not m.finished_at or not m.summary or not m.summary.get("n_scored"):
            continue
        cut = m.split_version or cut_of(domain, m.split, list(m.task_ids))
        runs.append(
            {
                "run_id": m.run_id,
                "agent": m.agent,
                "split": m.split,
                "cut": cut,
                "of": cut_size(domain, m.split, cut),
                "passed": m.summary["passed"],
                "n": m.summary["n_scored"],
                "model": m.model,
                "effort": m.agent_effort,
            }
        )

    def diagnosis(path: Any) -> Any:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    versions = [
        {
            "name": v.name,
            "model": v.config.model,
            "effort": v.config.effort,
            "diagnosis": diagnosis(v.path / "diagnosis.json"),
        }
        for v in list_versions(domain)
    ]
    return version_history(domain, versions, runs, read_ledger(domain), read_registry(domain))
