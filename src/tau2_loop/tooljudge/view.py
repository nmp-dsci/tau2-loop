"""What the viewer shows of the tool judge, from committed files only.

On one conversation: J0's labels (`data/judge/<domain>.json`), J1's golden answer
(`data/judge/<domain>_gold.jsonl`, with the person's checks once frozen) and the newest J2 replay's
verdicts (`judge_runs/<id>/verdicts.jsonl`). Each part is None until its slice has run, so the
Trace page grows with the slices and never shows a placeholder.

Across the domain (`overview`): the label counts, the golden answers' summary, and every replay
scored against the golden answers as they stand now, which is what Optimise's judge view draws.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

from tau2_loop.tooljudge import gold, labels, replay


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
    }
