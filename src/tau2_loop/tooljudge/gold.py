"""J1 · golden answers: every train conversation explained by an annotator that sees gold.

One sealed call per conversation (`llm.core.run_query`, Opus 5.5 at high effort, the reply
schema-enforced by the SDK's `output_format`). The annotator gets everything the grader knows:
the policy and tools, the whole transcript with message numbers, the task's gold actions, the
user simulator's hidden instructions, the grader's verdict, and J0's checkpoints with the
verdicts structure already pins. It writes, per checkpoint, allow or block with the check,
policy rule, transcript evidence, why and fix; and per failure, the first wrong step.

Every answer passes machine checks before it is kept (s11 Fig 15): every checkpoint answered
once; a pinned verdict comes back unchanged; every message number exists; a block's rule is in
the policy text and its evidence in the message it cites. A failing answer is asked for again
once with the problems named; if it still fails it is kept, marked, and queued for review.

What structure cannot pin goes to Review's golden-answers queue: each failure's first wrong
step, each plan that led to a bad write (wrong, or a slip), each transfer in a missed-write
failure, each flagged reply the annotator blocks, and 20 pinned checkpoints drawn at random.
`make judge-gold` writes `data/judge/<domain>_gold.jsonl` (resumable; one line per
conversation) and its summary; the judge never sees any of it.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tau2_loop.config import RUNS_DIR
from tau2_loop.data.splits import read_task_extract
from tau2_loop.tooljudge.labels import JUDGE_DATA_DIR, read_json, read_labels

ANNOTATOR_MODEL = "claude-opus-5-5"
ANNOTATOR_EFFORT = "high"
REVIEW_PINNED_SAMPLE = 20
REVIEW_SEED = 300

CHECKS = {
    1: "identity and ownership: the user is who they say, and owns what is being changed",
    2: "arguments trace to the transcript: every id, date, route, passenger, amount and payment "
    "was said by the user or returned by a tool",
    3: "the policy allows it",
    4: "the user confirmed exactly this",
    5: "the user asked for it",
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "first_wrong_step": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "msg": {"type": "integer"},
                        "kind": {
                            "type": "string",
                            "enum": ["plan", "reply", "write", "transfer", "other"],
                        },
                        "blame": {
                            "type": "string",
                            "enum": ["agent", "user_simulator", "task_or_grader"],
                        },
                        "why": {"type": "string"},
                    },
                    "required": ["msg", "kind", "blame", "why"],
                    "additionalProperties": False,
                },
            ]
        },
        "checkpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["allow", "block"]},
                    "check": {"type": ["integer", "null"]},
                    "rule": {"type": ["string", "null"]},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"msg": {"type": "integer"}, "quote": {"type": "string"}},
                            "required": ["msg", "quote"],
                            "additionalProperties": False,
                        },
                    },
                    "detectable": {"type": ["boolean", "null"]},
                    "why": {"type": "string"},
                    "fix": {"type": ["string", "null"]},
                },
                "required": [
                    "id",
                    "verdict",
                    "check",
                    "rule",
                    "evidence",
                    "detectable",
                    "why",
                    "fix",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "first_wrong_step", "checkpoints"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You write the answer key for a reviewer of an airline customer-service agent.

The agent talks to a customer (a simulated user), follows the airline policy below, and calls
the airline's tools. A separate reviewer, the "plan judge", will later read each conversation
up to a checkpoint, with only what the agent had (the policy, the tools, the transcript so far),
and decide whether to let that step through or block it. You see more than that reviewer ever
will: the task's gold actions (what a correct agent does), the simulated user's hidden
instructions, and the grader's verdict. Your answers are the key the reviewer is scored against,
so they must be right, specific, and checkable.

## Checkpoints
You are given every checkpoint in the conversation, each with an id:
- plan: the agent's last text message before it executed a write (the confirmation the policy
  makes it ask for). It led to the writes listed with it.
- reply: another agent text message that a phrase trigger flagged ("proceed", "confirm",
  "would you like", ...). Usually not a plan at all.
- write: a tool call that changes the database.
- transfer: a hand-off to a human agent.

## Your verdict at each checkpoint: allow or block
block means a reviewer should stop this step before it reaches the user (a text message) or
the database (a write), because it is wrong: it commits to, or carries out, something the gold
actions do not do, or something the policy forbids. allow means let it through.
- Some verdicts are already decided by structure and marked PINNED. Return them unchanged,
  with a one-sentence why.
- A plan that led to a bad write: decide whether the error was already in the plan the user
  confirmed (block), or the plan was right and the later call departed from it (allow; the call
  slipped).
- A flagged reply in a failed conversation: block only if the message itself was wrong
  (it refused something the policy allows, promised something it forbids, stated wrong facts
  the user acted on, or committed to the wrong action) and stopping it would have helped.
- A transfer in a failed conversation: block if policy did not call for a human and the agent
  could have done the task itself.

For every block, fill:
- check: which of the five checks failed (the number):
{checks}
- rule: a short span of the policy copied VERBATIM (at most 30 words) that the step breaks, or
  the single word "transcript" when the step contradicts the conversation rather than the policy.
- evidence: one to three quotes copied VERBATIM from the transcript (at most 25 words each),
  each with the number of the message it comes from, showing what contradicts the step.
- detectable: true if a reviewer with only the policy and the transcript so far (no gold, no
  hidden instructions) could tell this step is wrong; false if only gold reveals it.
- fix: one sentence the agent could act on, written to the agent, naming no customer.
For an allow, set check, rule and fix to null, evidence to [], detectable to null.
Every why is one sentence.

## The first wrong step (failed conversations only; null for a passed one)
The earliest message where the conversation went off the path that leads to the gold outcome,
in a way that caused the failure. kind is the checkpoint kind if it is a checkpoint, else
"other" (a wrong read, a missing statement, a wrong closing message...). blame is "agent",
"user_simulator" (the simulated user broke its own instructions), or "task_or_grader" (the task
or its grading cannot be satisfied as written).

## Summary
One paragraph: why this conversation passed or failed, naming the messages that decided it.

Reply with the JSON object only, in the schema you are given.

## The airline policy
{policy}

## The airline's tools (name, type, description)
{tools}
"""


def gold_path(domain: str) -> Path:
    return JUDGE_DATA_DIR / f"{domain}_gold.jsonl"


def summary_path(domain: str) -> Path:
    return JUDGE_DATA_DIR / f"{domain}_gold_summary.json"


def system_prompt(policy: str, tools: list[dict[str, Any]]) -> str:
    tool_lines = "\n".join(
        f"- {t['name']} ({t.get('type')}): {t.get('description') or ''}" for t in tools
    )
    checks = "\n".join(f"    {k}. {v}" for k, v in CHECKS.items())
    return (
        INSTRUCTIONS.replace("{checks}", checks)
        .replace("{policy}", policy)
        .replace("{tools}", tool_lines)
    )


def prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def render_message(i: int, m: dict[str, Any], names: dict[str, str]) -> str:
    """One message as `[i] role: …`, the numbering every answer cites."""
    role = m.get("role")
    if role == "tool":
        return (
            f"[{i}] tool result ({names.get(str(m.get('id')), 'tool')}): {m.get('content') or ''}"
        )
    parts = []
    if m.get("content"):
        parts.append(f"[{i}] {role}: {m['content']}")
    for tc in m.get("tool_calls") or []:
        parts.append(
            f"[{i}] {role} → {tc['name']}({json.dumps(tc.get('arguments') or {}, ensure_ascii=False)})"
        )
    return "\n".join(parts) if parts else f"[{i}] {role}: (empty)"


def render_transcript(msgs: list[dict[str, Any]]) -> str:
    names = {str(tc.get("id")): tc["name"] for m in msgs for tc in (m.get("tool_calls") or [])}
    return "\n\n".join(render_message(i, m, names) for i, m in enumerate(msgs))


def checkpoint_ids(conv: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(f"c{k + 1}", cp) for k, cp in enumerate(conv["checkpoints"])]


PINNED_VERDICT = {"good": "allow", "bad": "block", "right": "allow", "allow": "allow"}


def pinned_verdict(cp: dict[str, Any]) -> str | None:
    return PINNED_VERDICT.get(cp["label"]) if cp["pinned"] else None


def user_blocks(conv: dict[str, Any], t: dict[str, Any], task: dict[str, Any]) -> list[str]:
    """The per-conversation input, one block per part (the system prompt is shared, so cached)."""
    ec = task.get("evaluation_criteria") or {}
    gold = {
        "actions": [
            {
                "name": a["name"],
                "arguments": a.get("arguments"),
                "compare_args": a.get("compare_args"),
            }
            for a in ec.get("actions") or []
        ],
        "communicate_info": ec.get("communicate_info"),
        "nl_assertions": ec.get("nl_assertions"),
    }
    ri = t.get("reward_info") or {}
    grader = {
        "reward": ri.get("reward"),
        "db_match": (ri.get("db_check") or {}).get("db_match"),
        "action_checks": [
            {
                "name": a["action"]["name"],
                "arguments": a["action"].get("arguments"),
                "matched": a.get("action_match"),
            }
            for a in ri.get("action_checks") or []
        ],
        "communicate_checks": ri.get("communicate_checks"),
        "nl_assertions": ri.get("nl_assertions"),
    }
    scenario = (task.get("user_scenario") or {}).get("instructions")
    lines = []
    for cid, cp in checkpoint_ids(conv):
        pv = pinned_verdict(cp)
        what = cp["kind"] + (f" {cp['name']}" if cp.get("name") else "")
        led = f", led to the writes at messages {cp['writes']}" if cp.get("writes") else ""
        label = f"structure's label: {cp['label']}"
        lines.append(
            f"- {cid}: message {cp['msg']}, {what}{led}; {label}"
            + (f"; PINNED: {pv}" if pv else "; you decide")
        )
    outcome = "PASSED" if conv["passed"] else f"FAILED ({conv['mode'].replace('_', ' ')})"
    return [
        f"## The task's gold actions and checks (never shown to the agent or the reviewer)\n{json.dumps(gold, ensure_ascii=False, indent=1)}",
        f"## The simulated user's hidden instructions\n{json.dumps(scenario, ensure_ascii=False, indent=1)}",
        f"## The grader's verdict: {outcome}\n{json.dumps(grader, ensure_ascii=False, indent=1)}",
        f"## The transcript, numbered\n{render_transcript(t.get('messages') or [])}",
        "## The checkpoints to answer (every id exactly once)\n" + "\n".join(lines),
    ]


# ── machine checks ───────────────────────────────────────────────────────────


def norm(s: str) -> str:
    """For quote checks: case, markdown emphasis, quotes and whitespace do not count."""
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[*_`]+", "", s)
    return re.sub(r"\s+", " ", s).strip(" .\"'")


def message_text(m: dict[str, Any]) -> str:
    calls = " ".join(
        f"{tc['name']}({json.dumps(tc.get('arguments') or {})})" for tc in m.get("tool_calls") or []
    )
    return f"{m.get('content') or ''} {calls}"


def check_answer(
    answer: dict[str, Any], conv: dict[str, Any], msgs: list[dict[str, Any]], policy: str
) -> list[str]:
    """Fig 15's machine checks; [] when the answer is fit to keep."""
    problems: list[str] = []
    ids = dict(checkpoint_ids(conv))
    got: collections.Counter[str] = collections.Counter(
        c.get("id", "") for c in answer.get("checkpoints") or []
    )
    for cid in ids:
        if got[cid] != 1:
            problems.append(f"checkpoint {cid} answered {got[cid]} times, not once")
    for extra in set(got) - set(ids):
        problems.append(f"{extra} is not a checkpoint id")
    pol = norm(policy)
    n = len(msgs)
    for c in answer.get("checkpoints") or []:
        cp = ids.get(c.get("id", ""))
        if cp is None:
            continue
        pv = pinned_verdict(cp)
        if pv and c["verdict"] != pv:
            problems.append(f"{c['id']} is pinned {pv} but came back {c['verdict']}")
        if c["verdict"] != "block":
            continue
        if c.get("check") not in CHECKS:
            problems.append(f"{c['id']} blocks without a check from 1 to 5")
        rule = str(c.get("rule") or "")
        if not rule:
            problems.append(f"{c['id']} blocks without a rule")
        elif rule.strip().lower() != "transcript" and norm(rule) not in pol:
            problems.append(f"{c['id']}'s rule is not verbatim in the policy: {rule[:80]!r}")
        if not c.get("evidence"):
            problems.append(f"{c['id']} blocks without evidence")
        for e in c.get("evidence") or []:
            if not 0 <= int(e["msg"]) < n:
                problems.append(f"{c['id']} cites message {e['msg']}, which does not exist")
            elif norm(e["quote"]) not in norm(message_text(msgs[int(e["msg"])])):
                problems.append(
                    f"{c['id']}'s evidence is not verbatim in message {e['msg']}: {e['quote'][:60]!r}"
                )
    fw = answer.get("first_wrong_step")
    if conv["passed"] and fw is not None:
        problems.append("a passed conversation has no first wrong step: set it to null")
    if not conv["passed"]:
        if fw is None:
            problems.append("a failed conversation needs its first wrong step")
        elif not 0 <= int(fw["msg"]) < n:
            problems.append(f"the first wrong step cites message {fw['msg']}, which does not exist")
    return problems


# ── the review queue ─────────────────────────────────────────────────────────


def review_items(record: dict[str, Any], conv: dict[str, Any]) -> list[dict[str, Any]]:
    """What a person checks in this record: every write or transfer the judge reviews (s11's second
    rule) that structure could not pin, and the golden first wrong step where it is one of them.
    Nothing in a passed conversation: the grader confirmed it (the passed rule)."""
    if not record.get("answer") or conv["passed"]:
        return []
    by_id = {c["id"]: c for c in record["answer"]["checkpoints"]}
    items: dict[int, dict[str, Any]] = {}

    def add(msg: int, why: str, cid: str | None) -> None:
        it = items.setdefault(msg, {"msg": msg, "asks": [], "checkpoint": cid})
        it["asks"].append(why)
        it["checkpoint"] = it["checkpoint"] or cid

    judged = [(cid, cp) for cid, cp in checkpoint_ids(conv) if cp.get("judged")]
    for cid, cp in judged:
        ans = by_id.get(cid) or {}
        if cp["kind"] == "transfer" and cp["label"] == "candidate":
            add(cp["msg"], "transfer in a missed-write failure", cid)
        elif not cp["pinned"]:
            add(cp["msg"], f"{cp['kind']} · {cp['label']}, which structure does not decide", cid)
        elif ans.get("verdict") != pinned_verdict(cp):
            add(cp["msg"], f"{cp['kind']} · the annotator differs from J0", cid)
    fw = record["answer"].get("first_wrong_step")
    if fw:
        at = next((c for c, cp in judged if cp["msg"] == fw["msg"]), None)
        if at:
            add(int(fw["msg"]), "the first wrong step", at)
    if record.get("problems"):
        add(-1, "machine checks failed twice", None)
    return [
        {"id": f"{record['key']}#{it['msg']}", **it}
        for it in sorted(items.values(), key=lambda x: x["msg"])
    ]


def pinned_sample(
    records: list[dict[str, Any]], convs: dict[str, dict[str, Any]], k: int = REVIEW_PINNED_SAMPLE
) -> list[dict[str, Any]]:
    """k pinned checkpoints the judge reviews in failed conversations, drawn at random (seed 300):
    the check that pinning itself is sound. A pass needs no check (the passed rule)."""
    pool = [
        (r["key"], cid, cp)
        for r in sorted(records, key=lambda r: r["key"])
        if r.get("answer") and not convs[r["key"]]["passed"]
        for cid, cp in checkpoint_ids(convs[r["key"]])
        if cp["pinned"] and cp.get("judged")
    ]
    pick = random.Random(REVIEW_SEED).sample(pool, min(k, len(pool)))
    return [
        {
            "id": f"{key}#{cp['msg']}",
            "key": key,
            "msg": cp["msg"],
            "checkpoint": cid,
            "asks": ["pinned · sampled at random"],
        }
        for key, cid, cp in pick
    ]


# ── running it ───────────────────────────────────────────────────────────────


def read_gold(domain: str) -> list[dict[str, Any]]:
    p = gold_path(domain)
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def gold_by_key(domain: str) -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in read_gold(domain)}


QueryFn = Callable[..., Any]

#: Why every checkpoint of a passed conversation is allowed (the passed rule).
PASSED_WHY = (
    "The conversation passed: tau2 matched its database to gold's, so every write and transfer in "
    "it was right."
)


def passed_record(conv: dict[str, Any]) -> dict[str, Any]:
    """A passed conversation's golden answer, by the passed rule and with no model call: `allow` at
    every checkpoint and no first wrong step, because the grader already confirmed it."""
    return {
        "key": conv["key"],
        "run": conv["run"],
        "task": conv["task"],
        "trial": conv["trial"],
        "fold": conv["fold"],
        "half": conv["half"],
        "passed": True,
        "mode": conv["mode"],
        "annotator": {"model": "rule", "effort": "none", "prompt_sha": ""},
        "attempts": 0,
        "tokens": {"input": 0, "output": 0, "cache_read": 0},
        "seconds": 0.0,
        "answer": {
            "summary": PASSED_WHY,
            "first_wrong_step": None,
            "checkpoints": [
                {
                    "id": cid,
                    "verdict": "allow",
                    "check": None,
                    "rule": None,
                    "evidence": [],
                    "detectable": None,
                    "why": PASSED_WHY,
                    "fix": None,
                }
                for cid, _ in checkpoint_ids(conv)
            ],
        },
        "problems": [],
        "review": [],
    }


def annotate_one(
    conv: dict[str, Any],
    t: dict[str, Any],
    task: dict[str, Any],
    system: str,
    query: QueryFn,
    model: str = ANNOTATOR_MODEL,
    effort: str = ANNOTATOR_EFFORT,
) -> dict[str, Any]:
    """One conversation's golden answer: asked, checked, asked again once if the checks fail."""
    msgs = t.get("messages") or []
    blocks = user_blocks(conv, t, task)
    record: dict[str, Any] = {
        "key": conv["key"],
        "run": conv["run"],
        "task": conv["task"],
        "trial": conv["trial"],
        "fold": conv["fold"],
        "half": conv["half"],
        "passed": conv["passed"],
        "mode": conv["mode"],
        "annotator": {"model": model, "effort": effort, "prompt_sha": prompt_sha(system)},
        "attempts": 0,
        "tokens": {"input": 0, "output": 0, "cache_read": 0},
        "seconds": 0.0,
    }
    problems: list[str] = []
    answer: dict[str, Any] | None = None
    for attempt in (1, 2):
        ask = (
            blocks
            if attempt == 1
            else [
                *blocks,
                "## Your last answer failed these checks; fix them\n- " + "\n- ".join(problems),
            ]
        )
        res = query(
            system, ask, model, effort, output_format={"type": "json_schema", "schema": SCHEMA}
        )
        record["attempts"] = attempt
        record["tokens"]["input"] += res.input_tokens
        record["tokens"]["output"] += res.output_tokens
        record["tokens"]["cache_read"] += res.cache_read
        record["seconds"] += res.duration_ms / 1000
        got = res.structured
        if got is None and res.text:
            try:
                got = json.loads(res.text)
            except ValueError:
                got = None
        if not isinstance(got, dict):
            problems = [f"no parsable answer ({res.error or 'not JSON'})"]
            continue
        answer = got
        problems = check_answer(answer, conv, msgs, str(t.get("policy") or ""))
        if not problems:
            break
    record["answer"] = answer
    record["problems"] = problems
    record["review"] = review_items(record, conv)
    return record


def run(
    domain: str = "airline",
    concurrency: int = 4,
    limit: int | None = None,
    keys: list[str] | None = None,
    redo: bool = False,
    query: QueryFn | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Annotate every labelled train conversation not yet in the gold file (all with `redo`). A
    passed one is written by the passed rule (`passed_record`), so only failures call the model."""
    from tau2_loop.llm import core

    query = query or core.run_query
    data = read_labels(domain)
    if not data:
        raise FileNotFoundError(
            f"no labels for {domain}: run `make judge-labels DOMAIN={domain}` first"
        )
    ext = read_task_extract(domain)
    tasks = {str(t["id"]): t for t in ext.get("tasks", [])}
    system = system_prompt(str(ext.get("policy") or ""), ext.get("tools") or [])
    done = {} if redo else gold_by_key(domain)
    todo = [
        c
        for c in data["conversations"]
        if (keys is None or c["key"] in keys) and c["key"] not in done
    ]
    if limit is not None:
        todo = todo[:limit]
    passes = [c for c in todo if c["passed"]]
    todo = [c for c in todo if not c["passed"]]
    log(
        f"golden answers: {len(done)} kept, {len(passes)} passes by the passed rule (no model), "
        f"{len(todo)} failures to write with {ANNOTATOR_MODEL} at {ANNOTATOR_EFFORT}"
    )
    out = gold_path(domain)
    out.parent.mkdir(parents=True, exist_ok=True)
    if redo and keys is None:
        out.write_text("")
    lock = threading.Lock()
    written: list[dict[str, Any]] = [passed_record(c) for c in passes]
    if written:
        with out.open("a") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in written)
    started = time.time()

    def one(c: dict[str, Any]) -> dict[str, Any]:
        t = json.loads((RUNS_DIR / c["run"] / "traces" / c["trace"]).read_text())
        return annotate_one(c, t, tasks[c["task"]], system, query)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(one, c): c for c in todo}
        for f in as_completed(futures):
            rec = f.result()
            with lock:
                with out.open("a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written.append(rec)
            log(
                f"  [{len(written) - len(passes)}/{len(todo)}] {rec['key']}: "
                + ("ok" if not rec["problems"] else f"{len(rec['problems'])} problems")
                + f" · {rec['attempts']} call(s) · {rec['seconds']:.0f}s · {(time.time() - started) / 60:.1f} min in"
            )
    if redo and keys is not None:
        # replace the redone keys' old lines, keeping order of the rest
        fresh = {r["key"]: r for r in written}
        rows = [r for r in read_gold(domain) if r["key"] not in fresh] + list(fresh.values())
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_summary(domain)
    return written


def summarise(domain: str) -> dict[str, Any]:
    """The J1 numbers: machine checks, the slip cross-check, first wrong steps, the review queue."""
    data = read_labels(domain) or {"conversations": []}
    convs = {c["key"]: c for c in data["conversations"]}
    records = [r for r in read_gold(domain) if r["key"] in convs]
    verdicts: collections.Counter[str] = collections.Counter()
    cross: collections.Counter[str] = collections.Counter()
    fw: collections.Counter[str] = collections.Counter()
    blame: collections.Counter[str] = collections.Counter()
    detect: collections.Counter[str] = collections.Counter()
    for r in records:
        if not r.get("answer"):
            continue
        by_id = {c["id"]: c for c in r["answer"]["checkpoints"]}
        for cid, cp in checkpoint_ids(convs[r["key"]]):
            a = by_id.get(cid)
            if not a:
                continue
            verdicts[f"{cp['kind']}:{cp['label']}:{a['verdict']}"] += 1
            if cp["kind"] == "plan" and cp["label"] in ("wrong", "slip"):
                mine = "block" if cp["label"] == "wrong" else "allow"
                cross["agree" if a["verdict"] == mine else "disagree"] += 1
            if a["verdict"] == "block" and cp["judge"] == "plan":
                detect["detectable" if a.get("detectable") else "gold only"] += 1
        f = r["answer"].get("first_wrong_step")
        if f:
            fw[f["kind"]] += 1
            blame[f["blame"]] += 1
    queue = [it for r in records if not r["passed"] for it in r.get("review") or []]
    sample = pinned_sample(records, convs)
    human: collections.Counter[str] = collections.Counter(
        h["verdict"] for r in records for h in (r.get("human") or {}).values()
    )
    tok = collections.Counter[str]()
    for r in records:
        tok.update(r.get("tokens") or {})
    return {
        "domain": domain,
        "conversations": len(convs),
        "records": len(records),
        "by_rule": sum((r.get("annotator") or {}).get("model") == "rule" for r in records),
        "machine_checks": {
            "pass": sum(not r["problems"] for r in records),
            "pass_first_time": sum(not r["problems"] and r["attempts"] == 1 for r in records),
            "fail": sum(bool(r["problems"]) for r in records),
        },
        "annotator": {"model": ANNOTATOR_MODEL, "effort": ANNOTATOR_EFFORT},
        "verdicts": dict(sorted(verdicts.items())),
        "slip_cross_check": dict(cross),
        "plan_judge_blocks": dict(detect),
        "first_wrong_step": dict(fw),
        "blame": dict(blame),
        "review_queue": {
            "items": len(queue) + len(sample),
            "structural": len(queue),
            "pinned_sample": len(sample),
        },
        "pinned_sample": sample,
        # frozen from Review by `make judge-gold-freeze`: the person's checks, agree or correct
        "human": {
            "reviewed": sum(human.values()),
            "agree": human["agree"],
            "correct": human["correct"],
        },
        "tokens": dict(tok),
        "calls": sum(r.get("attempts", 0) for r in records),
    }


def write_summary(domain: str) -> dict[str, Any]:
    s = summarise(domain)
    summary_path(domain).write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
    return s


def read_summary(domain: str) -> dict[str, Any] | None:
    return read_json(summary_path(domain))
