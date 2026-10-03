"""A person's check of the golden answers (s11 J1): the queue, the verdicts, and the freeze.

The passed rule (2026-10-02): a passed conversation is confirmed by the grader. tau2 matched its
database to gold's, so every write and transfer in it was right and its golden answer is `allow`
at each, whatever the annotator said; no person checks it. A person confirms each failed one: the
golden answer at every write and transfer, so the first call the judge should block (or none), and
a failed conversation with no write or transfer as a whole (`WHOLE`), since the judge is never
called in it.

The queue is every case structure could not pin (`gold.review_items`) plus 20 pinned ones drawn
at random, from failed conversations only. A person agrees with the annotator or corrects it, and
can take a check back (`withdraw`); rows go to the central Postgres (`tau2_loop.gold_review`,
append-only, newest current), because a person typed them after the run. `freeze` copies the
current rows into the committed gold file as each record's `human` answers, and the person's answer
stands wherever it differs: `effective_verdicts` is what J2 and J3 score against.
"""

from __future__ import annotations

import json
from typing import Any

from tau2_loop.data import pg
from tau2_loop.tooljudge import gold
from tau2_loop.tooljudge.labels import read_labels

VERDICTS = ("agree", "correct", "withdraw")
#: The message number of a check on the conversation as a whole, not one message.
WHOLE = -1
_COLUMNS = "id, domain, item_id, conv_key, msg, verdict, correction, note, author, annotator_sha, created_at"


def _row(r: tuple[Any, ...]) -> dict[str, Any]:
    out = dict(zip([c.strip() for c in _COLUMNS.split(",")], r, strict=True))
    out["created_at"] = str(out["created_at"])
    return out


def add(
    domain: str,
    item_id: str,
    verdict: str,
    correction: dict[str, Any] | None = None,
    note: str = "",
    author: str = "",
    annotator_sha: str = "",
) -> dict[str, Any]:
    """Append one check. Raises ValueError on a bad verdict, correction or item id."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    if verdict == "correct" and not correction:
        raise ValueError("a correction says what the answer should be")
    if verdict == "withdraw" and correction:
        raise ValueError("a withdrawal takes a check back; it carries no correction")
    from psycopg.types.json import Jsonb

    key, _, msg = item_id.rpartition("#")
    if not key or not msg.lstrip("-").isdigit():
        raise ValueError("an item id is <run>/<task>/t<n>#<message>")
    with pg.connect() as con:
        row = con.execute(
            f"insert into {pg.SCHEMA}.gold_review "
            "(domain, item_id, conv_key, msg, verdict, correction, note, author, annotator_sha) "
            f"values (%s, %s, %s, %s, %s, %s, %s, %s, %s) returning {_COLUMNS}",
            (
                domain,
                item_id,
                key,
                int(msg),
                verdict,
                Jsonb(correction or {}),
                note,
                author,
                annotator_sha,
            ),
        ).fetchone()
    assert row is not None
    return _row(row)


def current(domain: str) -> dict[str, dict[str, Any]]:
    """The newest check per item; an item whose newest row withdraws its check has none."""
    with pg.connect_ro() as con:
        rows = con.execute(
            f"select distinct on (item_id) {_COLUMNS} from {pg.SCHEMA}.gold_review "
            "where domain = %s order by item_id, created_at desc, id desc",
            (domain,),
        ).fetchall()
    return {r["item_id"]: r for r in map(_row, rows) if r["verdict"] != "withdraw"}


def current_conversation(domain: str, conv_key: str) -> dict[str, dict[str, Any]]:
    """The newest check per message of one conversation, keyed by message number."""
    with pg.connect_ro() as con:
        rows = con.execute(
            f"select distinct on (item_id) {_COLUMNS} from {pg.SCHEMA}.gold_review "
            "where domain = %s and conv_key = %s order by item_id, created_at desc, id desc",
            (domain, conv_key),
        ).fetchall()
    return {str(r["msg"]): r for r in map(_row, rows) if r["verdict"] != "withdraw"}


def history(domain: str, item_id: str) -> list[dict[str, Any]]:
    with pg.connect_ro() as con:
        rows = con.execute(
            f"select {_COLUMNS} from {pg.SCHEMA}.gold_review where domain = %s and item_id = %s "
            "order by created_at desc, id desc",
            (domain, item_id),
        ).fetchall()
    return [_row(r) for r in rows]


# ── the queue as the viewer reads it ─────────────────────────────────────────


def queue(domain: str) -> list[dict[str, Any]]:
    """Every review item with what the person needs to judge it, in conversation order."""
    data = read_labels(domain) or {"conversations": []}
    convs = {c["key"]: c for c in data["conversations"]}
    records = gold.read_gold(domain)
    by_key = {r["key"]: r for r in records}
    items = [it | {"key": r["key"]} for r in records for it in r.get("review") or []]
    seen = {it["id"] for it in items}
    items += [it for it in gold.pinned_sample(records, convs) if it["id"] not in seen]
    out = []
    for it in items:
        rec, conv = by_key.get(it["key"]), convs.get(it["key"])
        # the passed rule: the grader confirmed a pass, so nothing in it waits for a person
        if rec is None or conv is None or conv["passed"]:
            continue
        out.append(describe(it, rec, conv))
    return sorted(out, key=lambda x: (x["key"], x["msg"]))


def item(domain: str, item_id: str) -> dict[str, Any] | None:
    """One case: the queue's, or a case opened from its conversation at any other write or transfer
    the judge reviews, so a person can correct a golden answer wherever they read it, or the
    conversation as a whole (`#-1`). A text reply is never a case: the judge is not called there
    (s11's second rule)."""
    found = next((i for i in queue(domain) if i["id"] == item_id), None)
    if found is not None:
        return found
    key, _, msg = item_id.rpartition("#")
    if not key or not msg.lstrip("-").isdigit():
        return None
    m = int(msg)
    rec = gold.gold_by_key(domain).get(key)
    conv = next(
        (
            c
            for c in (read_labels(domain) or {"conversations": []})["conversations"]
            if c["key"] == key
        ),
        None,
    )
    if rec is None or conv is None or not rec.get("answer"):
        return None
    if m == WHOLE:
        whole = {
            "id": item_id,
            "msg": m,
            "checkpoint": None,
            "asks": ["the conversation as a whole"],
        }
        return describe(whole, rec, conv)
    if m < 0:
        return None
    cid = next(
        (c for c, cp in gold.checkpoint_ids(conv) if cp["msg"] == m and cp.get("judged")), None
    )
    if cid is None:
        return None
    opened = {"id": item_id, "msg": m, "checkpoint": cid, "asks": ["opened from the conversation"]}
    return describe(opened, rec, conv)


def describe(it: dict[str, Any], rec: dict[str, Any], conv: dict[str, Any]) -> dict[str, Any]:
    ids = dict(gold.checkpoint_ids(conv))
    answer = rec.get("answer") or {}
    cid = it.get("checkpoint")
    cp = ids.get(cid) if cid else None
    ans = next((c for c in answer.get("checkpoints") or [] if c["id"] == cid), None)
    fw = answer.get("first_wrong_step")
    return {
        "id": it["id"],
        "key": rec["key"],
        "run": rec["run"],
        "task": rec["task"],
        "trial": rec["trial"],
        "msg": it["msg"],
        "asks": it["asks"],
        "fold": rec["fold"],
        "half": rec["half"],
        "passed": rec["passed"],
        "mode": rec["mode"],
        "checkpoint": {"id": cid, **cp} if cp else None,
        "answer": ans,
        "first_wrong_step": fw if fw and int(fw["msg"]) == it["msg"] else None,
        "is_first_wrong": bool(fw and int(fw["msg"]) == it["msg"]),
        "summary": answer.get("summary"),
        "problems": rec.get("problems") or [],
        "annotator_sha": (rec.get("annotator") or {}).get("prompt_sha", ""),
    }


def item_messages(domain: str, item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The checkpoint message and every message its evidence cites, whole, by number."""
    from tau2_loop.config import RUNS_DIR

    data = read_labels(domain) or {"conversations": []}
    conv = next((c for c in data["conversations"] if c["key"] == item["key"]), None)
    if conv is None:
        return {}
    msgs = (
        json.loads((RUNS_DIR / conv["run"] / "traces" / conv["trace"]).read_text()).get("messages")
        or []
    )
    wanted = {item["msg"]} | {
        int(e["msg"]) for e in ((item.get("answer") or {}).get("evidence") or [])
    }
    names = {str(tc.get("id")): tc["name"] for m in msgs for tc in (m.get("tool_calls") or [])}
    return {
        str(i): {"role": msgs[i].get("role"), "text": gold.render_message(i, msgs[i], names)}
        for i in sorted(wanted)
        if 0 <= i < len(msgs)
    }


# ── the freeze ───────────────────────────────────────────────────────────────


def freeze(domain: str) -> dict[str, Any]:
    """Copy every current check into the gold file (`human`, keyed by message) and count agreement."""
    cur = current(domain)
    records = gold.read_gold(domain)
    for r in records:
        mine = {
            str(v["msg"]): {
                k: v[k] for k in ("verdict", "correction", "note", "author", "created_at")
            }
            for v in cur.values()
            if v["conv_key"] == r["key"]
        }
        if mine:
            r["human"] = mine
        else:
            r.pop("human", None)
    gold.gold_path(domain).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    human: dict[str, Any] = gold.write_summary(domain)["human"]
    return human


def effective_verdicts(record: dict[str, Any], conv: dict[str, Any]) -> dict[str, str]:
    """Each checkpoint id's golden verdict, with the person's correction where there is one; in a
    passed conversation every write and transfer is `allow` (the passed rule)."""
    ids = gold.checkpoint_ids(conv)
    out = {c["id"]: c["verdict"] for c in (record.get("answer") or {}).get("checkpoints") or []}
    for cid, cp in ids:
        h = (record.get("human") or {}).get(str(cp["msg"]))
        if h and h["verdict"] == "correct" and h["correction"].get("verdict") in ("allow", "block"):
            out[cid] = h["correction"]["verdict"]
        if conv["passed"] and cp.get("judged"):
            out[cid] = "allow"
    return out


def effective_first_wrong(record: dict[str, Any]) -> dict[str, Any] | None:
    fw = (record.get("answer") or {}).get("first_wrong_step")
    newest_first = sorted(
        (record.get("human") or {}).values(),
        key=lambda h: str(h.get("created_at") or ""),
        reverse=True,
    )
    for h in newest_first:
        c = h.get("correction") or {}
        if h["verdict"] == "correct" and "first_wrong_msg" in c:
            m = c["first_wrong_msg"]
            return (
                None
                if m is None
                else {
                    **(fw or {}),
                    "msg": int(m),
                    "kind": c.get("first_wrong_kind") or (fw or {}).get("kind", "other"),
                }
            )
    return fw
