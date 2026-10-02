"""J2 · first verdicts: a judge version replayed on every checkpoint it would see live, then scored.

The replay walks each labelled train conversation and asks the judge at every checkpoint the plan
trigger flags (456 on airline train, v1 onwards), in order, with the real conversation up to that point, plus
J0's synthetic positives (a right plan with one detail changed). Each conversation's checkpoints go
to one worker in order, so the later ones read the earlier prefix from the prompt cache.

A replay is a folder, `judge_runs/<ts>_<domain>_<judge>_<split>/`, immutable once finished like a
run folder: `run.json` (the judge's ref, fingerprint, model, effort, threshold, and the hashes of
the labels and gold it was scored on), `verdicts.jsonl` (one line per item) and `summary.json`.

Scoring is s11 §9's. Per conversation, in order: a pass is interrupted if the judge blocks
anything in it; a wrong-plan conversation (its golden first wrong step is a plan the trigger
flags) is stopped if the judge's first block is that plan. Balanced accuracy is the mean of the
passes left alone and the wrong-plan conversations stopped. Per checkpoint: right plans blocked,
flagged replies in passes blocked, wrong plans blocked, the confusion against the golden verdicts,
and reason agreement (both block, the same check, an evidence message in common). Every number is
given for the read half, the gate half and all train; the gate half is the one J3 is judged on.
"""

from __future__ import annotations

import collections
import hashlib
import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau2_loop.config import ROOT, RUNS_DIR, settings
from tau2_loop.data.splits import read_task_extract
from tau2_loop.tooljudge import core, prompt
from tau2_loop.tooljudge.gold import checkpoint_ids, gold_by_key, gold_path
from tau2_loop.tooljudge.labels import labels_path, read_labels
from tau2_loop.tooljudge.review import effective_first_wrong, effective_verdicts

JUDGE_RUNS_DIR = ROOT / "judge_runs"
HALVES = ("read", "gate", "all")


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.is_file() else None


def items(domain: str) -> list[dict[str, Any]]:
    """Every live checkpoint of every labelled train conversation, then the synthetic positives."""
    data = read_labels(domain)
    if not data:
        raise FileNotFoundError(
            f"no labels for {domain}: run `make judge-labels DOMAIN={domain}` first"
        )
    out = []
    for c in data["conversations"]:
        for cid, cp in checkpoint_ids(c):
            if cp["live"]:
                out.append(
                    {
                        "item": "checkpoint",
                        "key": c["key"],
                        "msg": cp["msg"],
                        "cid": cid,
                        "kind": cp["kind"],
                        "label": cp["label"],
                        "fold": c["fold"],
                        "half": c["half"],
                        "passed": c["passed"],
                    }
                )
    for s in data["synthetic"]:
        out.append(
            {
                "item": "synthetic",
                "key": s["key"],
                "msg": s["msg"],
                "synthetic": s["id"],
                "kind": "plan",
                "label": "wrong",
                "what": s["kind"],
                "fold": s["fold"],
                "half": s["half"],
                "passed": None,
            }
        )
    return out


def run_replay(
    domain: str = "airline",
    judge_name: str = "j1",
    split: str = "train",
    concurrency: int = 4,
    limit: int | None = None,
    resume: str | None = None,
    query: core.QueryFn | None = None,
    log: Callable[[str], None] = print,
    track: bool = True,
) -> Path:
    """Replay a judge version over the domain's train checkpoints; returns the replay folder."""
    if split != "train":
        raise ValueError("the judge is calibrated on train only; test is reported once, by J6")
    judge = prompt.load(domain, judge_name)
    ext = read_task_extract(domain)
    policy = str(ext.get("policy") or "")
    system = prompt.system_prompt(judge, policy, ext.get("tools") or [])
    convs = {c["key"]: c for c in (read_labels(domain) or {})["conversations"]}
    synth = {s["id"]: s for s in (read_labels(domain) or {})["synthetic"]}
    todo = items(domain)
    if limit is not None:
        keys = list(dict.fromkeys(i["key"] for i in todo))[:limit]
        todo = [i for i in todo if i["key"] in keys]

    if resume:
        out = JUDGE_RUNS_DIR / resume
        meta = json.loads((out / "run.json").read_text())
        if meta.get("finished_at"):
            raise ValueError(f"{resume} is finished, and a finished replay is never rewritten")
    else:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = JUDGE_RUNS_DIR / f"{ts}_{domain}_{judge_name}_{split}"
        out.mkdir(parents=True)
        meta = {
            "replay_id": out.name,
            "domain": domain,
            "split": split,
            "judge": judge.ref,
            "name": judge.name,
            "fingerprint": judge.fingerprint,
            "model": judge.model,
            "effort": judge.effort,
            "threshold": judge.threshold,
            "structured": judge.structured,
            "labels_sha": _sha(labels_path(domain)),
            "gold_sha": _sha(gold_path(domain)),
            "code_sha": settings().code_sha,
            "items": len(todo),
            "limit": limit,
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
        }
        (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    vpath = out / "verdicts.jsonl"
    done = {(v["item"], v["key"], v["msg"], v.get("synthetic")) for v in read_verdicts(out)}
    todo = [i for i in todo if (i["item"], i["key"], i["msg"], i.get("synthetic")) not in done]
    by_conv: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for i in todo:
        by_conv[i["key"]].append(i)
    log(
        f"replay {out.name}: {judge.ref} ({judge.model}, {judge.effort}) on {len(todo)} items in {len(by_conv)} conversations"
    )

    lock = threading.Lock()
    count = [0]
    started = time.time()

    def one_conversation(key: str) -> None:
        c = convs[key]
        msgs = (
            json.loads((RUNS_DIR / c["run"] / "traces" / c["trace"]).read_text()).get("messages")
            or []
        )
        for it in sorted(by_conv[key], key=lambda x: (x["item"] != "checkpoint", x["msg"])):
            text = (
                synth[it["synthetic"]]["text"]
                if it["item"] == "synthetic"
                else str(msgs[it["msg"]].get("content") or "")
            )
            v = core.judge_reply(judge, system, policy, msgs, it["msg"], text, query=query)
            line = {**it, **v}
            with lock:
                with vpath.open("a") as fh:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                count[0] += 1
                if count[0] % 25 == 0 or count[0] == len(todo):
                    log(f"  {count[0]}/{len(todo)} · {(time.time() - started) / 60:.1f} min")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(one_conversation, list(by_conv)))

    summary = write_summary(out, meta)
    meta["finished_at"] = datetime.now(UTC).isoformat()
    meta["gold_sha"] = _sha(gold_path(domain))
    (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    if track:
        from tau2_loop.tooljudge import tracking

        log(f"MLflow: {tracking.log_replay(out, summary)}")
    return out


def write_summary(out: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """`summary.json`: the scores against the golden answers as they stand when it is written.

    The verdicts are the record and never change; a summary is the scorer's reading of them. When
    the scorer is fixed or the gold is frozen again, `make judge-score` rewrites it and notes why in
    `run.json`, and the viewer always scores live."""
    keys = (
        "replay_id",
        "domain",
        "split",
        "judge",
        "name",
        "fingerprint",
        "model",
        "effort",
        "threshold",
        "structured",
    )
    summary = {k: meta[k] for k in keys}
    summary.update(score(str(meta["domain"]), read_verdicts(out)))
    summary["gold_sha"] = _sha(gold_path(str(meta["domain"])))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def rescore(replay_id: str, why: str) -> dict[str, Any]:
    """Rewrite a finished replay's summary with today's scorer and gold; the reason goes in run.json."""
    out = JUDGE_RUNS_DIR / replay_id
    meta = json.loads((out / "run.json").read_text())
    summary = write_summary(out, meta)
    meta.setdefault("rescored", []).append(
        {"at": datetime.now(UTC).isoformat(), "why": why, "gold_sha": summary["gold_sha"]}
    )
    (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    return summary


def read_verdicts(out: Path) -> list[dict[str, Any]]:
    p = out / "verdicts.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def list_replays(domain: str | None = None) -> list[dict[str, Any]]:
    """Every replay folder's run.json, newest first."""
    if not JUDGE_RUNS_DIR.is_dir():
        return []
    out = []
    for d in sorted(JUDGE_RUNS_DIR.iterdir(), reverse=True):
        if (d / "run.json").is_file():
            m = json.loads((d / "run.json").read_text())
            if domain is None or m.get("domain") == domain:
                out.append(m)
    return out


# ── scoring ──────────────────────────────────────────────────────────────────


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def score(
    domain: str, verdicts: list[dict[str, Any]], gold: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """s11 §9's numbers for one replay, against the golden answers (the person's corrections win)."""
    data = read_labels(domain) or {"conversations": []}
    gold = gold_by_key(domain) if gold is None else gold
    v_by = {(v["key"], v["msg"]): v for v in verdicts if v["item"] == "checkpoint"}
    rows: list[dict[str, Any]] = []
    conv_rows: list[dict[str, Any]] = []
    for c in data["conversations"]:
        rec = gold.get(c["key"])
        eff = effective_verdicts(rec, c) if rec else {}
        answers = {a["id"]: a for a in ((rec or {}).get("answer") or {}).get("checkpoints") or []}
        fw = effective_first_wrong(rec) if rec else c["first_wrong"]
        live = [(cid, cp) for cid, cp in checkpoint_ids(c) if cp["live"]]
        first_block, replayed = None, 0
        for cid, cp in live:
            v = v_by.get((c["key"], cp["msg"]))
            if v is None:
                continue
            replayed += 1
            g, a = eff.get(cid), answers.get(cid) or {}
            raw = v.get("raw") or {}
            same_reason = (
                g == "block"
                and v["verdict"] == "block"
                and raw.get("check") == a.get("check")
                and bool(
                    {int(e["msg"]) for e in raw.get("evidence") or []}
                    & {int(e["msg"]) for e in a.get("evidence") or []}
                )
            )
            rows.append(
                {
                    "key": c["key"],
                    "run": c["run"],
                    "task": c["task"],
                    "trial": c["trial"],
                    "msg": cp["msg"],
                    "half": c["half"],
                    "passed": c["passed"],
                    "kind": cp["kind"],
                    "label": cp["label"],
                    "golden": g,
                    "judge": v["verdict"],
                    "same_reason": same_reason,
                    "error": bool(v.get("error")),
                    "detectable": a.get("detectable"),
                    "judge_check": raw.get("check"),
                    "golden_check": a.get("check"),
                    "judge_why": raw.get("why"),
                    "golden_why": a.get("why"),
                    "confidence": raw.get("confidence"),
                    "noted": bool(v.get("note")),
                }
            )
            if first_block is None and v["verdict"] == "block":
                first_block = cp["msg"]
        live_msgs = {cp["msg"] for _, cp in live}
        conv_rows.append(
            {
                "key": c["key"],
                "half": c["half"],
                "passed": c["passed"],
                "replayed": replayed,
                "live": len(live),
                "first_block": first_block,
                "first_wrong": fw,
                "wrong_plan": bool(
                    not c["passed"]
                    and fw
                    and fw.get("kind") == "plan"
                    and int(fw["msg"]) in live_msgs
                ),
                "basis": "gold" if rec else "labels",
            }
        )
    syn = [v for v in verdicts if v["item"] == "synthetic"]
    # a conversation counts once every checkpoint it has was replayed; one with none (the trigger
    # never fired) counts too, as left alone, unless this replay was cut short
    seen = {v["key"] for v in verdicts}
    full = all(c["key"] in seen for c in conv_rows if c["live"])
    for c in conv_rows:
        c["scored"] = c["replayed"] == c["live"] and (c["live"] > 0 or full)
    out: dict[str, Any] = {"scores": {}, "conversations": conv_rows, "rows": rows}
    for half in HALVES:
        hr = [r for r in rows if half == "all" or r["half"] == half]
        hc = [c for c in conv_rows if (half == "all" or c["half"] == half) and c["scored"]]
        hs = [v for v in syn if half == "all" or v["half"] == half]
        passes = [c for c in hc if c["passed"]]
        wp = [c for c in hc if c["wrong_plan"]]
        interrupted = sum(c["first_block"] is not None for c in passes)
        stopped = sum(c["first_block"] == int(c["first_wrong"]["msg"]) for c in wp)
        early = sum(
            c["first_block"] is not None and c["first_block"] < int(c["first_wrong"]["msg"])
            for c in wp
        )
        hg = [r for r in hr if r["golden"] in ("allow", "block")]
        tp = sum(r["golden"] == "block" and r["judge"] == "block" for r in hg)
        fn = sum(r["golden"] == "block" and r["judge"] == "allow" for r in hg)
        fp = sum(r["golden"] == "allow" and r["judge"] == "block" for r in hg)
        tn = sum(r["golden"] == "allow" and r["judge"] == "allow" for r in hg)
        right = [r for r in hr if r["kind"] == "plan" and r["label"] in ("right", "slip")]
        wrong = [r for r in hr if r["kind"] == "plan" and r["label"] == "wrong"]
        replies = [r for r in hr if r["kind"] == "reply" and r["passed"]]
        both = [r for r in hg if r["golden"] == "block" and r["judge"] == "block"]
        det = [r for r in hg if r["golden"] == "block" and r["detectable"]]
        left = (len(passes) - interrupted) / len(passes) if passes else None
        caught = stopped / len(wp) if wp else None
        out["scores"][half] = {
            "balanced_accuracy": round((left + caught) / 2, 4)
            if left is not None and caught is not None
            else None,
            "passes_interrupted": _rate(interrupted, len(passes)),
            "wrong_plans_stopped": _rate(stopped, len(wp)),
            "wrong_plans_blocked_early": _rate(early, len(wp)),
            "right_plans_blocked": _rate(sum(r["judge"] == "block" for r in right), len(right)),
            "pass_replies_blocked": _rate(
                sum(r["judge"] == "block" for r in replies), len(replies)
            ),
            "wrong_plans_blocked": _rate(sum(r["judge"] == "block" for r in wrong), len(wrong)),
            "synthetic_blocked": _rate(sum(v["verdict"] == "block" for v in hs), len(hs)),
            "golden_blocks_caught": _rate(tp, tp + fn),
            "golden_allows_blocked": _rate(fp, fp + tn),
            "detectable_blocks_caught": _rate(sum(r["judge"] == "block" for r in det), len(det)),
            "checkpoint_balanced_accuracy": round((tp / (tp + fn) + tn / (tn + fp)) / 2, 4)
            if tp + fn and tn + fp
            else None,
            "reason_agreement": _rate(sum(r["same_reason"] for r in both), len(both)),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "checkpoints": len(hr),
            "with_gold": len(hg),
            "failed_open": sum(r["error"] for r in hr),
            # raw blocks the decision rule turned into notes (confidence, or a rule not in the policy)
            "blocks_noted": _rate(sum(r["noted"] for r in hr), len(hr)),
            "noted_golden_blocks": _rate(
                sum(r["noted"] and r["golden"] == "block" for r in hr), sum(r["noted"] for r in hr)
            ),
        }
    tok: collections.Counter[str] = collections.Counter()
    for v in verdicts:
        tok.update(v.get("tokens") or {})
    out["tokens"] = dict(tok)
    out["items"] = {
        "checkpoints": sum(v["item"] == "checkpoint" for v in verdicts),
        "synthetic": len(syn),
    }
    out["gold_records"] = len(gold)
    out["gold_human"] = sum(len(r.get("human") or {}) for r in gold.values())
    return out


def latest(domain: str, judge: str | None = None) -> Path | None:
    """The newest finished replay folder of the domain (of one judge version, if given)."""
    for m in list_replays(domain):
        if (
            m.get("finished_at")
            and not m.get("limit")
            and (judge is None or m.get("name") == judge)
        ):
            return JUDGE_RUNS_DIR / str(m["replay_id"])
    return None
