"""The merge bench's eval (s30): what a merge wrote, scored before anything is played.

  E1 answer key   one grader call per version: does the job, as the answering agent reads it
                  (`library.steps_view`), lead the agent to each fact gold's filings need, and does
                  it state any wrong rule found in an earlier version (a trap)? It also lists
                  conflicts: two instructions for one case (E4's model half).
  E2 accounting   code: each old step that stood on its documents is kept, accounted for (covered,
                  replaced by a passage, removed by the audit, the reviewer or the ledger with a
                  reason) or dropped silently; how many of the old version's quotes survive.
  E3 sourced      one reviewer call per final head: the shared reviewer (`review.md`, the derived
                  standard) reads every step, not only the ones code picks.
  E4 structure    code: dangling step ids (D19), listed writes no step makes (D20), missing write
                  parameters (D9), whether a step files, size; plus E1's conflicts.

The answer key (`merge_bench/<domain>/eval/keys/<job>.json`) is drafted from train gold and read
here only: no merge prompt contains it (tests/test_merge_bench.py). The grader is checked first on
versions that already exist (`calibrate`), where s30 found by hand what each has and lacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau2_loop.config import ROOT
from tau2_loop.workflows import bench, library, provenance, rag_agent, rubric

Ask = Callable[[str], "str | None"]  # one fresh-context call: the message -> the reply
FACT_POINTS = {"right": 1.0, "partly": 0.5, "missing": 0.0, "wrong": 0.0}
CACHE = ROOT / "merge_runs" / ".eval_cache"  # one grader reply per (prompt, version), reused


def key(domain: str, job: str) -> dict[str, Any]:
    k: dict[str, Any] = json.loads(
        (bench.BENCH_DIR / domain / "eval" / "keys" / f"{job}.json").read_text()
    )
    return k


def _prompt(domain: str, name: str) -> str:
    return (bench.BENCH_DIR / domain / "eval" / name).read_text()


class Grader:
    """A grader call with its tag (s31, O8): the cache and the eval file of a grader other than
    bench.yaml's carry the tag, so one model's replies are never read as another's."""

    def __init__(self, ask: Ask, tag: str = "") -> None:
        self.ask, self.tag = ask, tag

    def __call__(self, text: str) -> str | None:
        return self.ask(text)


def live_ask(domain: str, grader: str = "") -> Grader:
    """The grader's model and effort from bench.yaml, on the subscription; `grader`
    ("<model>:<effort>", s31 O8) names another, which is then the tag."""
    from tau2_loop.llm import core, require_live

    require_live()
    g = bench.config(domain).get("grader") or {}
    model, effort = str(g.get("model", "opus")), str(g.get("effort", "high"))
    tag = ""
    if grader and grader != f"{model}:{effort}":
        model, _, effort = grader.partition(":")
        effort = effort or "high"
        tag = f"{model}-{effort}"
    resolved = core.resolve_model(model)

    def ask(text: str) -> str | None:
        ans = core.answer([{"role": "user", "content": text}], None, resolved, effort=effort)
        return str(ans.content) if ans.content is not None else None

    return Grader(ask, tag)


def tag_of(ask: Ask) -> str:
    return str(getattr(ask, "tag", "") or "")


def _cached(kind: str, text: str, ask: Ask, log: Path | None) -> str | None:
    h = hashlib.sha256(text.encode()).hexdigest()[:24]
    if tag_of(ask):
        kind = f"{kind}@{tag_of(ask)}"
    p = CACHE / kind / f"{h}.json"
    if p.is_file():
        return str(json.loads(p.read_text())["reply"])
    reply = ask(text)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"kind": kind, "at": datetime.now(UTC).isoformat(), "text": text, "reply": reply}
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    if log is not None:
        with log.open("a") as f:
            f.write(json.dumps({"kind": kind, "cache": p.name}) + "\n")
    return reply


# ── E1 ──


def grade_request(domain: str, wf: dict[str, Any], k: dict[str, Any]) -> str:
    facts = "\n".join(f"- {f['id']} ({f['field']}): {f['fact']}" for f in k["facts"])
    traps = "\n".join(f"- {t['id']}: {t['rule']}" for t in k["traps"])
    return _prompt(domain, "grade.md").format(
        job=wf.get("job", ""),
        workflow=json.dumps(library.steps_view(wf), ensure_ascii=False, indent=1),
        facts=facts,
        traps=traps,
    )


def parse_grade(text: str | None, k: dict[str, Any]) -> dict[str, Any] | None:
    from tau2_loop.workflows.rag_agent import _json_object

    obj = _json_object(text, lambda v: isinstance(v.get("facts"), list))
    if obj is None:
        return None
    facts = {
        str(f["id"]): f
        for f in obj["facts"]
        if isinstance(f, dict) and f.get("verdict") in FACT_POINTS
    }
    traps = {
        str(t["id"]): t for t in obj.get("traps") or [] if isinstance(t, dict) and "present" in t
    }
    if set(facts) != {f["id"] for f in k["facts"]} or set(traps) != {t["id"] for t in k["traps"]}:
        return None
    return {
        "facts": facts,
        "traps": traps,
        "conflicts": [c for c in obj.get("conflicts") or [] if isinstance(c, dict)],
        "score": round(sum(FACT_POINTS[f["verdict"]] for f in facts.values()), 2),
        "of": len(facts),
        "traps_present": sorted(i for i, t in traps.items() if t["present"] is True),
    }


def e1(
    domain: str, wf: dict[str, Any], k: dict[str, Any], ask: Ask, log: Path | None = None
) -> dict[str, Any]:
    text = grade_request(domain, wf, k)
    for attempt in range(2):
        got = parse_grade(_cached("grade" if attempt == 0 else "grade_again", text, ask, log), k)
        if got is not None:
            return got
    return {"error": "the grader's reply was unreadable twice"}


# ── E2 ──


def e2(
    domain: str,
    old: dict[str, Any],
    new: dict[str, Any],
    record: dict[str, Any] | None,
    policy: str,
) -> dict[str, Any]:
    """What one merge did with the old version's steps that stood on their documents."""
    from tau2_loop.workflows import merge_opts

    sourced = merge_opts.sourced_steps(domain, old, policy)
    now = {str(s.get("id")) for s in new.get("steps") or [] if isinstance(s, dict)}
    rec = record or {}
    named: dict[str, str] = {}
    for a in merge_opts.accounts(rec.get("option")):
        named[str(a["step"])] = "covered" if a.get("covered_by") else "replaced by a passage"
    for j in (rec.get("audit") or {}).values():
        for x in j.get("removed") or []:
            if x.get("step"):
                named[str(x["step"])] = "removed by the audit"
    for x in (rec.get("post_audit") or {}).get("removed") or []:
        if x.get("step"):
            named[str(x["step"])] = "removed by the post-merge audit"
    for x in rec.get("settled") or []:
        w = str(x.get("what", ""))
        if w.startswith("step ") and "reviewer" in str(x.get("why", "")):
            named[w[5:]] = "removed by the reviewer"
    for x in rec.get("changelog") or []:  # the merge's own changelog, not code's lines
        w, why = str(x.get("what", "")), str(x.get("why", ""))
        if x.get("change") == "removed" and "logged by code" not in why:
            for sid in sourced:
                if sid in w.split() or w == f"step {sid}" or w == sid:
                    named.setdefault(sid, "removed by the merge with a reason")
    led = rec.get("ledger")
    if led:
        claims = {c["id"]: c for c in led.get("claims") or []}
        rejected = set(led.get("rejected") or {})
        carried = set(led.get("survive") or [])
        for sid in sourced:
            mine = [
                c
                for c in claims.values()
                if any(
                    o.startswith("L") and o.split(":", 1)[-1] == sid for o in c.get("origin", [])
                )
            ]
            if mine and all(c["id"] in carried | rejected for c in mine):
                named.setdefault(
                    sid,
                    "rewritten from its claims"
                    if any(c["id"] in carried for c in mine)
                    else "every claim rejected with a reason",
                )
    kept = [s for s in sourced if s in now]
    accounted: dict[str, int] = {}
    silent: list[str] = []
    for sid in sourced:
        if sid in now:
            continue
        if sid in named:
            accounted[named[sid]] = accounted.get(named[sid], 0) + 1
        else:
            silent.append(sid)
    out: dict[str, Any] = {
        "sourced": len(sourced),
        "kept": len(kept),
        "accounted": accounted,
        "silent": silent,
    }
    from tau2_loop.workflows.rag_agent import quotes

    def keyed(wf: dict[str, Any]) -> set[str]:
        return {
            rubric._squash(q["quote"])[: rubric.QUOTE_KEY]
            for q in quotes({"jobs": [wf]})
            if q.get("quote")
        }

    oq, nq = keyed(old), keyed(new)
    out["old_quotes"], out["old_quotes_kept"] = len(oq), len(oq & nq)
    return out


# ── E3 ──


def e3(
    domain: str, wf: dict[str, Any], ask: Ask, policy: str, log: Path | None = None
) -> dict[str, Any]:
    picks = [
        {"step": str(s["id"]), "flags": {"E3": "every step is read"}}
        for s in wf.get("steps") or []
        if isinstance(s, dict) and s.get("id")
    ]
    text = provenance.review_request(domain, _prompt_review(domain), wf, picks, policy)
    verdicts = provenance.parse_review(_cached("review_all", text, ask, log))
    no = sorted({v["step"] for v in verdicts if v["verdict"] == "no"})
    partly = sorted({v["step"] for v in verdicts if v["verdict"] == "partly"} - set(no))
    derived = sum(
        v["verdict"] == "entails" and v["why"].lower().startswith("derived") for v in verdicts
    )
    return {
        "steps": len(picks),
        "no": no,
        "partly": partly,
        "clean": len(picks) - len(no) - len(partly),
        "derived_clauses": derived,
        "verdicts": verdicts,
    }


def _prompt_review(domain: str) -> str:
    return (bench.BENCH_DIR / domain / "review.md").read_text()


# ── E4 ──

FILING = "file_debit_card_transaction_dispute_6281"


def e4(domain: str, wf: dict[str, Any], files: str = FILING) -> dict[str, Any]:
    from tau2_loop.workflows import toolspec

    steps = [s for s in wf.get("steps") or [] if isinstance(s, dict)]
    calls = [rubric.step_tool(wf, k) for k in range(len(steps))]
    return {
        "chars": len(json.dumps(wf, ensure_ascii=False)),
        "steps": len(steps),
        "rules": len(wf.get("rules") or []),
        "files": files in calls,
        "D9": [
            f"{m['step']}: {', '.join(m['missing'])}" for m in toolspec.missing_params(domain, wf)
        ],
        "D19": [f"{m['where']} → {m['ref']}" for m in rubric.dangling(wf)],
        "D20": rubric.uncalled_writes(domain, wf),
    }


# ── a run, the baselines, the calibration ──


def score_version(
    domain: str, wf: dict[str, Any], k: dict[str, Any], ask: Ask, log: Path | None = None
) -> dict[str, Any]:
    return {"E1": e1(domain, wf, k, ask, log), "E4": e4(domain, wf)}


def run_eval(run_dir: Path, ask: Ask, workers: int = 4) -> dict[str, Any]:
    """Every head of a bench run (E1, E4), each merge's accounting (E2), the last head's E3."""
    rec = json.loads((run_dir / "run.json").read_text())
    domain, job = rec["domain"], rec["job"]
    k = key(domain, job)
    system = (
        rag_agent.RAG_RUNS_DIR / str(bench.config(domain)["jobs"][job]["system_from"]) / "system.md"
    ).read_text()
    policy = bench.policy_of(system)
    hs = bench.heads(run_dir, job)
    log = run_dir / "eval_calls.jsonl"
    with ThreadPoolExecutor(workers) as ex:
        scored = list(ex.map(lambda h: score_version(domain, h["workflow"], k, ask, log), hs))
        last = ex.submit(e3, domain, hs[-1]["workflow"], ask, policy, log)
        e3_last = last.result()
    by_v = {h["version"]: h["workflow"] for h in hs}
    merges = []
    for m in rec["merges"]:
        out_p = run_dir / "sessions" / m["session"] / "output.json"
        o = json.loads(out_p.read_text()) if out_p.is_file() else {}
        record = next(
            (x for x in o.get("merge") or [] if job in [x.get("job"), *(x.get("into") or [])]), None
        )
        old = by_v.get(m["read"]) if m.get("read") is not None else None
        new = by_v.get(m["committed"]) if m.get("committed") else old
        merges.append(
            {
                "task": m["task"],
                "read": m["read"],
                "committed": m["committed"],
                "taken": m["taken"],
                "failed": (record or {}).get("failed") or [],
                "E2": e2(domain, old, new, record, policy) if old and new else None,
            }
        )
    out = {
        "run": rec["id"],
        "option": rec["option"],
        "trial": rec["trial"],
        "status": rec["status"],
        "heads": [
            {"version": h["version"], "task": h["task"], **s}
            for h, s in zip(hs, scored, strict=True)
        ],
        "merges": merges,
        "E3_last": e3_last,
        "evaluated": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    }
    if tag_of(ask):
        out["grader"] = tag_of(ask)
    (run_dir / eval_name(ask)).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return out


def eval_name(ask: Ask) -> str:
    """bench.yaml's grader writes eval.json; another writes eval@<tag>.json beside it."""
    return f"eval@{tag_of(ask)}.json" if tag_of(ask) else "eval.json"


BASELINES = {  # the versions that already exist: r17's seed, r18's head, r19's three
    "r17_v0": ("session", "20261009T102616Z_banking_knowledge_r8_task_084"),
    "r18_v1": ("r18", 1),
    "r19_v1": ("r19", 1),
    "r19_v2": ("r19", 2),
    "r19_v3": ("r19", 3),
}


def baseline_version(domain: str, job: str, where: tuple[str, Any]) -> dict[str, Any]:
    kind, ref = where
    if kind == "session":
        wf = library._session_job(str(ref), job)
        assert wf is not None
        return wf
    wf2: dict[str, Any] = json.loads(
        (ROOT / "rag_library" / domain / kind / "versions" / job / f"{ref}.json").read_text()
    )
    return wf2


def calibrate(domain: str, job: str, ask: Ask, workers: int = 3) -> dict[str, Any]:
    """s31 (O8): E1 alone on the versions the key has hand labels for, against those labels: what
    a new grader must reproduce before it scores a merge (Opus agreed on 20 of 22 cells)."""
    k = key(domain, job)
    names = [n for n in k["calibration"] if n in BASELINES]
    wfs = [baseline_version(domain, job, BASELINES[n]) for n in names]
    with ThreadPoolExecutor(workers) as ex:
        graded = dict(zip(names, ex.map(lambda wf: e1(domain, wf, k, ask), wfs), strict=True))
    return {"grader": tag_of(ask) or "bench.yaml", **calibration(k, graded)}


def baselines(domain: str, job: str, ask: Ask, out: Path, workers: int = 5) -> dict[str, Any]:
    """E1, E3 and E4 on the versions that already exist, and E1 against the hand labels."""
    k = key(domain, job)
    system = (
        rag_agent.RAG_RUNS_DIR / str(bench.config(domain)["jobs"][job]["system_from"]) / "system.md"
    ).read_text()
    policy = bench.policy_of(system)
    wfs = {n: baseline_version(domain, job, w) for n, w in BASELINES.items()}
    out.mkdir(parents=True, exist_ok=True)
    log = out / "eval_calls.jsonl"
    with ThreadPoolExecutor(workers) as ex:
        sc = dict(
            zip(
                wfs,
                ex.map(lambda wf: score_version(domain, wf, k, ask, log), wfs.values()),
                strict=True,
            )
        )
        e3s = dict(
            zip(wfs, ex.map(lambda wf: e3(domain, wf, ask, policy, log), wfs.values()), strict=True)
        )
    res = {n: {**sc[n], "E3": e3s[n]} for n in wfs}
    res["calibration"] = calibration(k, {n: r["E1"] for n, r in res.items() if n in wfs})
    (out / eval_name(ask)).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    return res


def calibration(k: dict[str, Any], graded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each hand label against the grader's call; missing and wrong count as one."""
    rows, agree = [], 0
    for name, cells in k["calibration"].items():
        if name == "about" or name not in graded or "error" in graded[name]:
            continue
        g = graded[name]
        for cid, want in cells.items():
            if cid.startswith("K"):
                got = g["facts"][cid]["verdict"]
                ok = got == want or {got, want} <= {"missing", "wrong"}
            else:
                got = "present" if g["traps"][cid]["present"] is True else "absent"
                ok = got == want
            agree += ok
            rows.append({"version": name, "cell": cid, "hand": want, "grader": got, "agree": ok})
    return {"agree": agree, "of": len(rows), "rows": rows}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tau2_loop.workflows.merge_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score a bench run (calls the grader)")
    r.add_argument("run_dir")
    b = sub.add_parser("baselines", help="score r17, r18 and r19's versions and calibrate")
    c = sub.add_parser("calibrate", help="E1 on the hand-labelled versions only (O8)")
    for s in (b, c):
        s.add_argument("--domain", default="banking_knowledge")
        s.add_argument("--job", default="filing_a_debit_card_transaction_dispute")
    for s in (r, b, c):
        s.add_argument("--grader", default="", help="<model>:<effort> in place of bench.yaml's")
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        print(json.dumps(calibrate(a.domain, a.job, live_ask(a.domain, a.grader)), indent=1))
    elif a.cmd == "run":
        d = Path(a.run_dir)
        rec = json.loads((d / "run.json").read_text())
        res = run_eval(d, live_ask(rec["domain"], a.grader))
        print(
            json.dumps(
                [
                    {
                        "v": h["version"],
                        "E1": h["E1"].get("score"),
                        "traps": h["E1"].get("traps_present"),
                    }
                    for h in res["heads"]
                ]
            )
        )
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        res = baselines(
            a.domain,
            a.job,
            live_ask(a.domain, a.grader),
            ROOT / "merge_runs" / f"{stamp}_{a.domain}_baselines",
        )
        print(json.dumps(res["calibration"], indent=1))


if __name__ == "__main__":
    main()
