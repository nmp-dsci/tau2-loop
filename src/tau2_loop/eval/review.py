"""The human verdict on a conversation the judge already scored.

τ²'s `nl_assertions` check is a model deciding whether a sentence is true of a
transcript, and it is the only check with that property: the database hash, the
expected actions and the things that must be said are all deterministic. So a
failed conversation is not automatically a bad agent, and without a human label
set there is no way to tell a prompt regression from a judge flake.

A review is one of three things — `agree` with the judge, `disagree` with it, or
`task_broken`, the task itself cannot be satisfied — plus a reason. Rows are
append-only: the newest per (run, task, trial) is current, and a changed mind
stays readable. `judge_reward` is copied at the time of review, so a later
re-score cannot quietly rewrite what was disagreed with.

This is the one thing in the project that cannot be a committed file, because a
person typed it after the run. Everything else stays under `runs/`.
"""

from __future__ import annotations

from typing import Any

from tau2_loop.data import pg

VERDICTS = ("agree", "disagree", "task_broken")

_COLUMNS = "id, run_id, task_id, trial, judge_reward, verdict, reason, author, code_sha, created_at"


def _row(r: tuple[Any, ...]) -> dict[str, Any]:
    keys = [c.strip() for c in _COLUMNS.split(",")]
    out = dict(zip(keys, r, strict=True))
    out["judge_reward"] = float(out["judge_reward"])
    out["created_at"] = str(out["created_at"])
    return out


def add(
    run_id: str,
    task_id: str,
    trial: int,
    judge_reward: float,
    verdict: str,
    reason: str = "",
    author: str = "",
    code_sha: str = "",
) -> dict[str, Any]:
    """Append one review. Raises ValueError on a verdict the table would refuse."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    with pg.connect() as con:
        row = con.execute(
            f"insert into {pg.SCHEMA}.review "
            "(run_id, task_id, trial, judge_reward, verdict, reason, author, code_sha) "
            f"values (%s, %s, %s, %s, %s, %s, %s, %s) returning {_COLUMNS}",
            (run_id, task_id, trial, judge_reward, verdict, reason, author, code_sha),
        ).fetchone()
    assert row is not None
    return _row(row)


def history(run_id: str, task_id: str, trial: int) -> list[dict[str, Any]]:
    """Every review of one conversation, newest first. The first row is current."""
    with pg.connect_ro() as con:
        rows = con.execute(
            f"select {_COLUMNS} from {pg.SCHEMA}.review "
            "where run_id = %s and task_id = %s and trial = %s order by created_at desc, id desc",
            (run_id, task_id, trial),
        ).fetchall()
    return [_row(r) for r in rows]


def current(run_id: str | None = None) -> dict[str, dict[str, Any]]:
    """The newest review per conversation, keyed `<task_id>/t<trial>` (a run's own ids).

    One query with DISTINCT ON rather than one per row: a full-split run is 456
    conversations and the Review tab lists them all.
    """
    where, args = ("where run_id = %s", (run_id,)) if run_id else ("", ())
    with pg.connect_ro() as con:
        rows = con.execute(
            f"select distinct on (run_id, task_id, trial) {_COLUMNS} "
            f"from {pg.SCHEMA}.review {where} "
            "order by run_id, task_id, trial, created_at desc, id desc",
            args,
        ).fetchall()
    out = {}
    for r in rows:
        row = _row(r)
        key = f"{row['task_id']}/t{row['trial']}"
        if run_id is None:
            key = f"{row['run_id']}/{key}"
        out[key] = row
    return out


def tally(run_id: str | None = None) -> dict[str, int]:
    """How many conversations carry each verdict, counting only the current review."""
    counts = dict.fromkeys(VERDICTS, 0)
    for row in current(run_id).values():
        counts[str(row["verdict"])] += 1
    return counts
