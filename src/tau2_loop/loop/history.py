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
"""

from __future__ import annotations

from typing import Any

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
        "model": r.get("model"),
        "effort": r.get("effort"),
    }


def _newest(
    runs: list[dict[str, Any]], agent: str, split: str, cut: int | None = None
) -> dict[str, Any] | None:
    xs = [
        r
        for r in runs
        if r["agent"] == agent and r["split"] == split and (cut is None or r.get("cut") == cut)
    ]
    return xs[-1] if xs else None


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
            made = {"kind": "optimise", "cycle": None, "source": None, "detail": "no ledger entry"}
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
                "fixed": len(o["fixed"]) if "fixed" in o else None,
                "broke": len(o["broken"]) if "broken" in o else None,
                "p": o.get("p_value"),
                "vs": vs,
                "held_title": any(r["version"] == name for r in reigns),
            }
        )

    champ = registry.get("champion") or {}
    return {"domain": domain, "champion": champ.get("agent"), "versions": out, "reigns": reigns}
