"""What the viewer shows of the tool judge, from committed files only.

On one conversation: J0's labels (`data/judge/<domain>.json`), J1's golden answer
(`data/judge/<domain>_gold.jsonl`, with the person's checks once frozen) and the newest J2 replay's
verdicts (`judge_runs/<id>/verdicts.jsonl`). Each part is None until its slice has run, so the
Trace page grows with the slices and never shows a placeholder.

Across the domain (`overview`): the label counts, the golden answers' summary, every replay scored
against the golden answers as they stand now, and J3's ledger and registry, which is what Optimise's
judge view draws.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

from tau2_loop.tooljudge import gold, labels, prompt, replay
from tau2_loop.tooljudge import loop as judge_loop


@functools.lru_cache(maxsize=8)
def _jsonl_by_key(path: Path, mtime_ns: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out.setdefault(row["key"], []).append(row)
    return out


def _by_key(path: Path) -> dict[str, list[dict[str, Any]]]:
    return _jsonl_by_key(path, path.stat().st_mtime_ns) if path.is_file() else {}


def shown_replay(domain: str) -> dict[str, Any] | None:
    """The replay the viewer draws: the newest finished full one, else the newest still running."""
    rs = replay.list_replays(domain)
    full = [m for m in rs if not m.get("limit")]
    return next((m for m in full if m.get("finished_at")), None) or (full[0] if full else None)


def conversation(domain: str, run: str, task: str, trial: int) -> dict[str, Any] | None:
    """The judge's evidence on one conversation, or None when it is not a labelled train one."""
    lab = labels.conversation_labels(domain, run, task, trial)
    if lab is None:
        return None
    found = _by_key(gold.gold_path(domain)).get(lab["key"])
    rec = copy.deepcopy(found[0]) if found else None
    if rec is not None:
        ids = dict(gold.checkpoint_ids(lab))
        for a in (rec.get("answer") or {}).get("checkpoints") or []:
            cp = ids.get(a["id"])
            a["msg"], a["kind"] = (cp["msg"], cp["kind"]) if cp else (None, None)
    verdicts = None
    m = shown_replay(domain)
    if m:
        rows = (
            _by_key(replay.JUDGE_RUNS_DIR / m["replay_id"] / "verdicts.jsonl").get(lab["key"]) or []
        )
        verdicts = {
            "replay_id": m["replay_id"],
            "judge": m["judge"],
            "model": m["model"],
            "threshold": m["threshold"],
            "finished": bool(m.get("finished_at")),
            "items": [r for r in rows if r["item"] == "checkpoint"],
            "synthetic": [r for r in rows if r["item"] == "synthetic"],
        }
    return {"labels": lab, "gold": rec, "verdicts": verdicts}


def overview(domain: str) -> dict[str, Any]:
    """Optimise's judge view: labels, golden answers, and each replay scored on today's gold."""
    data = labels.read_labels(domain)
    reps = []
    for m in replay.list_replays(domain):
        d = replay.JUDGE_RUNS_DIR / m["replay_id"]
        s = replay.score(domain, replay.read_verdicts(d))
        s.pop("conversations", None)
        rows = s.pop("rows")
        # where the judge and the golden answer differ: what J3's optimiser reads (read half only)
        s["disagreements"] = [
            r for r in rows if r["golden"] in ("allow", "block") and r["golden"] != r["judge"]
        ]
        reps.append({**m, **s})
    return {
        "domain": domain,
        "labels": {"summary": data["summary"], "folds": data["folds"], "halves": data["halves"]}
        if data
        else None,
        "gold": gold.summarise(domain) if gold.read_gold(domain) else None,
        "probe": labels.read_json(labels.JUDGE_DATA_DIR / "probe.json"),
        "replays": reps,
        "registry": judge_loop.read_registry(domain)
        if judge_loop.registry_path(domain).is_file()
        else None,
        "cycles": judge_loop.read_ledger(domain),
    }


def versions(domain: str) -> list[dict[str, Any]]:
    """Every judge version of the domain, its files whole: the Agent tab's view of the judge agent."""
    out: list[dict[str, Any]] = []
    root = prompt.JUDGES_DIR / domain
    if not root.is_dir():
        return out
    replays = replay.list_replays(domain)
    for kind_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for d in sorted(p for p in kind_dir.iterdir() if (p / "judge.md").is_file()):
            j = prompt.load(domain, d.name, kind_dir.name)
            out.append(
                {
                    "ref": j.ref,
                    "kind": j.kind,
                    "name": j.name,
                    "model": j.model,
                    "effort": j.effort,
                    "threshold": j.threshold,
                    "structured": j.structured,
                    "fingerprint": j.fingerprint,
                    "config": j.config,
                    "files": {"judge.md": j.rubric, "judge.yaml": (d / "judge.yaml").read_text()},
                    "replays": [m["replay_id"] for m in replays if m.get("judge") == j.ref],
                }
            )
    return out


def evals(domain: str) -> dict[str, Any] | None:
    """The LLM judge's eval set: every labelled train conversation, its checkpoints and golden answer.

    What Evals shows with the scope bar on the judge: the items it is scored on, the folds and
    halves that keep the judge loop honest, and the synthetic positives. None before J0 has run."""
    data = labels.read_labels(domain)
    if not data:
        return None
    by_key = _by_key(gold.gold_path(domain))
    rows = []
    for c in data["conversations"]:
        found = by_key.get(c["key"])
        rec = found[0] if found else None
        answers = {a["id"]: a for a in ((rec or {}).get("answer") or {}).get("checkpoints") or []}
        ids = gold.checkpoint_ids(c)
        live = [(cid, cp) for cid, cp in ids if cp["live"]]
        rows.append(
            {
                "key": c["key"],
                "run": c["run"],
                "version": c["version"],
                "model": c.get("model"),
                "task": c["task"],
                "trial": c["trial"],
                "fold": c["fold"],
                "half": c["half"],
                "passed": c["passed"],
                "mode": c["mode"],
                "suspect": c.get("suspect", False),
                "checkpoints": len(live),
                "plans": sum(cp["kind"] == "plan" for _, cp in live),
                "golden_blocks": sum(
                    (answers.get(cid) or {}).get("verdict") == "block" for cid, _ in live
                ),
                "first_wrong": ((rec or {}).get("answer") or {}).get("first_wrong_step"),
                "has_gold": rec is not None,
                "human": len((rec or {}).get("human") or {}),
            }
        )
    return {
        "domain": domain,
        "summary": data["summary"],
        "folds": data["folds"],
        "halves": data["halves"],
        "conversations": rows,
        "synthetic": [
            {
                k: s[k]
                for k in ("id", "key", "msg", "task", "fold", "half", "kind", "what", "change")
            }
            for s in data["synthetic"]
        ],
        "gold": gold.summarise(domain) if by_key else None,
    }
