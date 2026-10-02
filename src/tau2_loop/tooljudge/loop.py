"""J3 (s11 §7): the judge loop. An optimiser edits the plan judge's lessons; the gate half decides.

A cycle starts from the champion judge and its newest full replay. The read half's disagreements
with the golden answers (folds F1–F3), plus the read half's synthetic positives the judge let
through, go to a fenced Opus 5.5 session as a bundle: for each, the conversation up to the reply,
the reply, the judge's whole answer and the golden one. The session writes one file,
`changes.json`: lessons to add, edit or remove by number, and why. The harness applies them to the
champion's `judge.md`, where nothing above `## Lessons` can change, refuses a lesson that names a
customer, an id, a disagreement or a sentence of a conversation, replays the challenger on every
train checkpoint, and gates it on the gate half (F4–F5), which the optimiser never reads.

The gate pairs the two judges conversation by conversation, as §8 walks them: a pass is right when
the judge blocks nothing in it, and a failure whose golden first wrong step is a flagged plan is
right when the judge's first block is that plan. The challenger is promoted when its balanced
accuracy on the gate half rises, the paired test agrees (one-sided McNemar p < 0.05, or at least
`MIN_FIXED` fixes and no breaks: this gate is small, so the agent gate's 1 is too few), and it
loses no §9 bar the champion meets. Each cycle is a line in `judges/<domain>/plan/ledger.jsonl`;
`registry.json` names the champion.

The optimiser reads only what the bundle holds, the rubric and the policy. Beyond
`loop/guards.py`'s fence it may not read `judge_runs/` (every replay holds the gate half's
verdicts), `.lavish/` (the plan pages quote gate-half conversations) or the ledger (it lists the
gate half's fixes and breaks). It may not run a replay or call a model.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau2_loop.config import ROOT, RUNS_DIR
from tau2_loop.data.splits import read_task_extract
from tau2_loop.eval.compare import ALPHA, mcnemar_one_sided
from tau2_loop.llm import Effort, redact, redact_tree, require_live
from tau2_loop.loop.guards import leak_values, leaks
from tau2_loop.loop.optimiser import CONTEXT_DIR, _run_session, _write_context, tree_checksum
from tau2_loop.tooljudge import prompt, replay
from tau2_loop.tooljudge.gold import checkpoint_ids, gold_by_key, gold_path
from tau2_loop.tooljudge.labels import labels_path, read_labels

KIND = "plan"
OPTIMISER_MODEL = "opus"
OPTIMISER_EFFORT: Effort = "high"
MAX_LESSONS = 60  # §7
MAX_ADDED = 10  # per cycle: a cycle makes delta edits, never a new rubric (s08: ACE, RELAI)
LESSON_CHARS = 500
QUOTE_WORDS = 8  # a run this long shared with a conversation, and not in the policy, is a quote
MIN_FIXED = 2  # §7: at least 2 fixes and no breaks when the paired test cannot speak
MESSAGE_CHARS = 500  # each message of a bundled conversation, as the optimiser reads it
EXCERPT_CHARS = 5000  # each bundled conversation: about 75k tokens for the read half's 61
HIDDEN = ("judge_runs", ".lavish")
LESSONS_HEAD = "## Lessons"
LESSONS_INTRO = (
    "General rules learned from where this judge disagreed with the golden answers on train. Each "
    "applies to any customer and any reservation; apply them with the checks above."
)
NO_LESSONS = (
    "None yet. J3's loop adds numbered lessons here, each a general rule, never a customer, task "
    "or\nid."
)
# s11 §9's bars for the plan judge, read on the gate half: (direction, limit)
BARS: dict[str, tuple[str, float]] = {
    "balanced_accuracy": (">=", 0.73),
    "passes_interrupted": ("<=", 2 / 55),
    "wrong_plans_stopped": (">=", 0.5),
    "right_plans_blocked": ("<=", 2 / 41),
    "pass_replies_blocked": ("<=", 0.05),
    "wrong_plans_blocked": (">=", 0.6),
    "reason_agreement": (">=", 0.7),
}


# ── where a judge's lineage lives ────────────────────────────────────────────


def plan_dir(domain: str) -> Path:
    return prompt.JUDGES_DIR / domain / KIND


def ledger_path(domain: str) -> Path:
    return plan_dir(domain) / "ledger.jsonl"


def registry_path(domain: str) -> Path:
    return plan_dir(domain) / "registry.json"


def read_ledger(domain: str) -> list[dict[str, Any]]:
    p = ledger_path(domain)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def read_registry(domain: str) -> dict[str, Any]:
    """The champion and every version's origin; before the first cycle, j1, written by hand."""
    p = registry_path(domain)
    if p.is_file():
        reg: dict[str, Any] = json.loads(p.read_text())
        return reg
    return {
        "domain": domain,
        "kind": KIND,
        "champion": "j1",
        "versions": [{"name": "j1", "made": "by hand, s11 J2", "cycle": None, "verdict": None}],
    }


def hidden_paths(domain: str) -> list[Path]:
    """What the optimiser may not read beyond the fence: every replay, the plan pages, the ledger."""
    return [(ROOT / h).resolve() for h in HIDDEN] + [ledger_path(domain).resolve()]


def next_name(domain: str) -> str:
    """jN+1 over every folder, a rejected challenger's included, so a name is never reused."""
    nums = [
        int(m.group(1))
        for p in plan_dir(domain).iterdir()
        if (m := re.fullmatch(r"j(\d+)", p.name))
    ]
    return f"j{max(nums, default=0) + 1}"


# ── the rubric: everything above ## Lessons is fixed ─────────────────────────


def split_rubric(md: str) -> tuple[str, list[str]]:
    """The fixed head of a judge.md and its numbered lessons, one per line."""
    head, sep, tail = md.partition(f"\n{LESSONS_HEAD}\n")
    if not sep:
        raise ValueError("judge.md has no ## Lessons section")
    lessons = [m.group(1).strip() for m in re.finditer(r"(?m)^\d+\. (.+)$", tail)]
    return head, lessons


def render_rubric(head: str, lessons: list[str], name: str) -> str:
    head = re.sub(r"^# The plan judge · \S+", f"# The plan judge · {name}", head, count=1)
    body = (
        LESSONS_INTRO + "\n\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(lessons, 1))
        if lessons
        else NO_LESSONS
    )
    return f"{head}\n{LESSONS_HEAD}\n\n{body}\n"


def apply_changes(
    lessons: list[str], changes: dict[str, Any]
) -> tuple[list[str], dict[str, Any], list[str]]:
    """The champion's lessons with `changes` applied by number: (lessons, what changed, errors)."""
    errors: list[str] = []
    n = len(lessons)
    out: list[str | None] = list(lessons)
    edited: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    added: list[str] = []

    def number(e: Any) -> int | None:
        k = e.get("n") if isinstance(e, dict) else None
        if isinstance(k, int) and 1 <= k <= n:
            return k
        errors.append(f"lesson {k} does not exist: the champion has {n}")
        return None

    for e in changes.get("edit") or []:
        if (k := number(e)) is not None:
            text = str(e.get("lesson") or "").strip()
            out[k - 1] = text or None
            edited.append({"n": k, "from": lessons[k - 1], "to": text})
    for e in changes.get("remove") or []:
        if (k := number(e)) is not None:
            out[k - 1] = None
            removed.append({"n": k, "lesson": lessons[k - 1]})
    for e in changes.get("add") or []:
        text = str((e.get("lesson") if isinstance(e, dict) else e) or "").strip()
        if text:
            out.append(text)
            added.append(text)
    if not (edited or removed or added):
        errors.append("changes.json changes no lesson")
    if len(added) > MAX_ADDED:
        errors.append(f"adds {len(added)} lessons; a cycle adds at most {MAX_ADDED}")
    touched = {e["n"] for e in edited} | {e["n"] for e in removed}
    if n >= 4 and len(touched) > n / 2:
        errors.append(f"rewrites {len(touched)} of {n} lessons; a cycle makes delta edits")
    new = [t for t in out if t]
    return new, {"added": added, "edited": edited, "removed": removed}, errors


_RESERVATION = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b")
_FLIGHT = re.compile(r"\bHAT\d{3}\b")
_USER = re.compile(r"\b[a-z]+_[a-z]+_\d{3,}\b")
_DISAGREEMENT = re.compile(r"\b[dD]\d{2,}\b")


_SAME = {"hrs": "hours", "hr": "hour", "mins": "minutes", "min": "minute"}


def _words(text: str) -> list[str]:
    """Lower-case words, with the policy's abbreviations spelled out as an agent says them."""
    return [_SAME.get(w, w) for w in re.findall(r"[a-z0-9$']+", text.lower())]


def quotes(text: str, conversations: list[str], policy: str, k: int = QUOTE_WORDS) -> list[str]:
    """Runs of `k` words that `text` shares with exactly one conversation and the policy lacks.

    A run that several conversations share is the agent's stock wording, often the policy restated
    ("it was booked within the last 24 hours"); a run only one holds is that conversation's own."""
    pol = f" {' '.join(_words(policy))} "
    where: dict[str, int] = {}
    for s in conversations:
        w = _words(s)
        for g in {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}:
            where[g] = where.get(g, 0) + 1
    w = _words(text)
    grams = (" ".join(w[i : i + k]) for i in range(len(w) - k + 1))
    return [g for g in grams if where.get(g) == 1 and f" {g} " not in pol]


def lesson_problems(
    lessons: list[str], domain: str, conversations: list[str], policy: str
) -> list[str]:
    """Why a set of lessons may not ship: too many, too long, or anything from one conversation.
    `conversations` holds one text per conversation (`by_conversation`)."""
    out: list[str] = []
    if len(lessons) > MAX_LESSONS:
        out.append(f"{len(lessons)} lessons, over the {MAX_LESSONS} limit")
    values = leak_values(domain)
    for i, t in enumerate(lessons, 1):
        if len(t) > LESSON_CHARS:
            out.append(f"lesson {i} is {len(t)} characters, over {LESSON_CHARS}")
        for what, rx in (
            ("a reservation id", _RESERVATION),
            ("a flight number", _FLIGHT),
            ("a user id", _USER),
            ("a disagreement id", _DISAGREEMENT),
        ):
            if rx.search(t):
                out.append(f"lesson {i} names {what}")
        if leaks({"": t}, values):  # the value is a customer's, so the reason never repeats it
            out.append(f"lesson {i} names a customer from a task")
        if quotes(t, conversations, policy):
            out.append(f"lesson {i} quotes a conversation")
    return out


# ── the bundle: the read half, never the gate half ───────────────────────────


def _excerpt(msgs: list[dict[str, Any]], at: int, reply: str, keep: set[int]) -> str:
    """The conversation up to the reply as the judge saw it, condensed: every message cut to
    `MESSAGE_CHARS`, the messages either answer cites always kept, then the latest ones that fit in
    `EXCERPT_CHARS`, with a marker where earlier ones were left out. The reply is always whole."""
    lines = prompt.blocks(msgs, at, reply)
    head, body, tail = lines[0], lines[1:-1], lines[-1]
    cut = [x if len(x) <= MESSAGE_CHARS else x[:MESSAGE_CHARS] + " […]" for x in body]
    chosen = {i for i in keep if 0 <= i < len(cut)}
    room = EXCERPT_CHARS - sum(len(cut[i]) for i in chosen)
    for i in range(len(cut) - 1, -1, -1):
        if i in chosen:
            continue
        if len(cut[i]) > room:
            break
        chosen.add(i)
        room -= len(cut[i])
    out, gap = [head], 0
    for i, x in enumerate(cut):
        if i in chosen:
            if gap:
                out.append(f"[… {gap} earlier message{'s' if gap > 1 else ''} left out …]")
                gap = 0
            out.append(x)
        else:
            gap += 1
    if gap:
        out.append(f"[… {gap} message{'s' if gap > 1 else ''} left out …]")
    return "\n".join([*out, tail])


def _cited(*answers: dict[str, Any] | None) -> set[int]:
    return {int(e["msg"]) for a in answers for e in ((a or {}).get("evidence") or []) if "msg" in e}


def disagreements(domain: str, verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every read-half checkpoint where the judge and the golden answer differ, then every read-half
    synthetic positive it let through, each with the conversation up to the reply."""
    data = read_labels(domain) or {"conversations": [], "synthetic": []}
    gold = gold_by_key(domain)
    convs = {c["key"]: c for c in data["conversations"]}
    synth = {s["id"]: s for s in data["synthetic"]}
    v_by = {(v["key"], v["msg"]): v for v in verdicts if v["item"] == "checkpoint"}
    traces: dict[str, list[dict[str, Any]]] = {}

    def msgs_of(key: str) -> list[dict[str, Any]]:
        if key not in traces:
            c = convs[key]
            trace = json.loads((RUNS_DIR / c["run"] / "traces" / c["trace"]).read_text())
            traces[key] = trace.get("messages") or []
        return traces[key]

    out: list[dict[str, Any]] = []
    for r in replay.score(domain, verdicts, gold)["rows"]:
        if (
            r["half"] != "read"
            or r["golden"] not in ("allow", "block")
            or r["golden"] == r["judge"]
        ):
            continue
        c = convs[r["key"]]
        cid = next(i for i, cp in checkpoint_ids(c) if cp["msg"] == r["msg"] and cp["live"])
        answers = ((gold.get(r["key"]) or {}).get("answer") or {}).get("checkpoints") or []
        a: dict[str, Any] = next((x for x in answers if x["id"] == cid), {})
        v = v_by[(r["key"], r["msg"])]
        m = msgs_of(r["key"])
        out.append(
            {
                "item": "checkpoint",
                "key": r["key"],
                "msg": r["msg"],
                "kind": r["kind"],
                "label": r["label"],
                "passed": r["passed"],
                "direction": "false block" if r["judge"] == "block" else "miss",
                "golden": {
                    "verdict": r["golden"],
                    **{k: a.get(k) for k in ("check", "rule", "why", "fix", "evidence")},
                },
                "judge": {"verdict": v["verdict"], "note": v.get("note"), "raw": v.get("raw")},
                "excerpt": _excerpt(
                    m, r["msg"], str(m[r["msg"]].get("content") or ""), _cited(a, v.get("raw"))
                ),
            }
        )
    for v in verdicts:
        if v["item"] != "synthetic" or v["half"] != "read" or v["verdict"] == "block":
            continue
        s = synth[v["synthetic"]]
        out.append(
            {
                "item": "synthetic",
                "key": v["key"],
                "msg": v["msg"],
                "kind": "plan",
                "label": "wrong",
                "passed": None,
                "direction": "miss",
                "golden": {
                    "verdict": "block",
                    "check": s.get("check", 2),
                    "rule": "transcript",
                    "why": f"a right plan with one detail changed, {s['what']}: "
                    f"{s['change']['from']} became {s['change']['to'] or '(dropped)'}",
                    "fix": None,
                    "evidence": [],
                },
                "judge": {"verdict": v["verdict"], "note": v.get("note"), "raw": v.get("raw")},
                "excerpt": _excerpt(msgs_of(v["key"]), v["msg"], s["text"], _cited(v.get("raw"))),
            }
        )
    for i, d in enumerate(out, 1):
        d["id"] = f"d{i:02d}"
    return out


def by_conversation(dis: list[dict[str, Any]]) -> list[str]:
    """The bundle's text, one string per conversation: what the quote guard counts across."""
    out: dict[str, list[str]] = {}
    for d in dis:
        out.setdefault(d["key"], []).append(d["excerpt"])
    return ["\n".join(v) for v in out.values()]


def _short(text: Any, n: int = 220) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _answer(a: dict[str, Any]) -> str:
    raw = a.get("raw") or a
    parts = [f"verdict: {a.get('verdict')}"]
    for k in ("confidence", "check", "rule", "fix", "why"):
        if raw.get(k) not in (None, "", []):
            parts.append(f"{k}: {raw[k]}")
    for e in raw.get("evidence") or []:
        parts.append(f"evidence: [{e.get('msg')}] {e.get('quote')}")
    if a.get("note"):
        parts.append(f"turned into a note by the decision rule: {a['note']}")
    return "\n".join(f"- {p}" for p in parts)


def _rate(r: dict[str, Any]) -> str:
    return f"{r['k']} / {r['n']}" if r.get("n") else "—"


def write_bundle(
    ctx: Path, champion: prompt.JudgeVersion, dis: list[dict[str, Any]], read: dict[str, Any]
) -> None:
    """`.context/`: the read half's scores, an index of the disagreements, and one file each."""
    (ctx / "scores.md").write_text(
        f"# How {champion.name} scores on the conversations you may read (the read half)\n\n"
        f"- balanced accuracy: {read['balanced_accuracy']}\n"
        f"- passing conversations it interrupts (blocks anything in): {_rate(read['passes_interrupted'])}\n"
        f"- failures whose first wrong step is a plan, stopped at that plan: {_rate(read['wrong_plans_stopped'])}\n"
        f"- right plans blocked: {_rate(read['right_plans_blocked'])}\n"
        f"- flagged replies (not plans) blocked in passing conversations: {_rate(read['pass_replies_blocked'])}\n"
        f"- wrong plans blocked: {_rate(read['wrong_plans_blocked'])}\n"
        f"- golden blocks it caught, every checkpoint: {_rate(read['golden_blocks_caught'])}\n"
        f"- golden allows it blocked, every checkpoint: {_rate(read['golden_allows_blocked'])}\n"
        f"- when both block, same check and an evidence message in common: {_rate(read['reason_agreement'])}\n"
        f"- synthetic positives blocked: {_rate(read['synthetic_blocked'])}\n"
        f"- raw blocks the decision rule turned into notes: {_rate(read['blocks_noted'])}\n"
    )
    lines = [
        "# Where the judge and the golden answer disagree, on the read half",
        "",
        "Each line: id · file · the checkpoint · the conversation's outcome · golden verdict · the judge's.",
        "",
    ]
    for d in dis:
        outcome = "synthetic" if d["item"] == "synthetic" else "passed" if d["passed"] else "failed"
        raw = d["judge"].get("raw") or {}
        g = d["golden"]
        golden = f"golden {g['verdict']}" + (f" (check {g['check']})" if g.get("check") else "")
        judge = f"judge {d['judge']['verdict']}" + (
            " (a raw block, noted)" if d["judge"].get("note") else ""
        )
        conf = f", confidence {raw.get('confidence')}" if raw else ""
        lines.append(
            f"- {d['id']} · {CONTEXT_DIR}/{d['id']}.md · {d['kind']} ({d['label']}) · {outcome} · "
            f"{d['direction']}: {golden}, {judge}{conf}\n"
            f"  golden: {_short(g.get('why'))}\n  judge: {_short(raw.get('why'))}"
        )
        outcome_line = (
            "a synthetic positive: a right plan with one detail changed"
            if d["item"] == "synthetic"
            else f"in a conversation that {'passed' if d['passed'] else 'failed'}"
        )
        (ctx / f"{d['id']}.md").write_text(
            f"# {d['id']} · a {d['direction']}\n\n"
            f"The checkpoint: message {d['msg']}, a {d['kind']} (J0's label: {d['label']}), {outcome_line}.\n\n"
            f"{d['excerpt']}\n\n"
            f"## The judge's answer ({champion.name})\n{_answer(d['judge'])}\n\n"
            f"## The golden answer\n{_answer(d['golden'])}\n"
        )
    (ctx / "disagreements.md").write_text("\n".join(lines) + "\n")


CHANGES_SCHEMA = """{
  "diagnosis": "a paragraph: the patterns behind the disagreements, and which you chose to fix",
  "patterns": [
    {"pattern": "one line", "direction": "false block | miss | wrong reason", "disagreements": ["d03", "d17"]}
  ],
  "add": [{"lesson": "the lesson's text", "because": ["d03", "d17"]}],
  "edit": [{"n": 2, "lesson": "the new text", "because": ["d05"]}],
  "remove": [{"n": 4, "because": ["d08"]}],
  "expect": "which disagreements these changes should fix, and which kind of reply must stay allowed"
}"""


def build_prompt(
    domain: str,
    champion: prompt.JudgeVersion,
    new_name: str,
    dis: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> str:
    n_fb = sum(d["direction"] == "false block" for d in dis)
    n_fb_pass = sum(d["direction"] == "false block" and d["passed"] for d in dis)
    n_miss = sum(d["direction"] == "miss" for d in dis)
    n_syn = sum(d["item"] == "synthetic" for d in dis)
    _, lessons = split_rubric(champion.rubric)
    past = (
        "\n".join(
            f"- cycle {h['cycle']}: {h['challenger']['name']} added {len(h['lessons']['added'])}, "
            f"edited {len(h['lessons']['edited'])}, removed {len(h['lessons']['removed'])} lessons; "
            f"{h['verdict']}"
            for h in history
        )
        or "- none: this is the first cycle"
    )
    where = f"judges/{domain}/{KIND}/{new_name}"
    return f"""You are improving an LLM judge. The plan judge `{champion.name}` reads each reply an airline
customer-service agent is about to send, before the customer sees it, and says allow or block. A block
makes the agent rewrite the reply. You may change one thing: the numbered lessons at the end of its
rubric. Its model ({champion.model}, {champion.effort} effort), its decision rule and every line above
`## Lessons` stay as they are.

## What the judge is for, and how it is scored
- Interrupting a conversation that was going to succeed is the costliest mistake: a block on a right
  reply can push the agent off a correct path. The bar is at most 2 of 55 passing conversations with
  any block in them.
- The gain is stopping a wrong plan before the customer agrees to it: a failed conversation whose first
  wrong step is a plan, blocked at that plan.
- The score is balanced accuracy over conversations: the mean of the share of passing conversations
  left alone and the share of wrong-plan failures stopped at the plan. Reason agreement (the same check
  as the golden answer) is reported beside it.
- The golden answers were written by a stronger model that saw each task's expected actions. Treat
  them as right.

## The decision rule (not yours to change)
A raw block stands only when its confidence is at least {champion.threshold} and its `rule` is a span
copied verbatim from the policy, or the word `transcript`. Otherwise the reply goes through and the
block is kept as a note. A block that paraphrases or elides the policy (with "…") is lost this way.

## Read
- `judges/{domain}/{KIND}/{champion.name}/judge.md`: the rubric, the system prompt the judge gets before
  the policy and the tools. It has {len(lessons)} lessons now.
- `{where}/{CONTEXT_DIR}/policy.md` and `tools.json`: what the judge is given.
- `{where}/{CONTEXT_DIR}/scores.md`: how `{champion.name}` scores on the conversations you may read.
- `{where}/{CONTEXT_DIR}/disagreements.md`: all {len(dis)} disagreements on those conversations, one
  line each: {n_fb} false blocks ({n_fb_pass} of them in conversations that passed), {n_miss - n_syn}
  misses of a golden block, and {n_syn} synthetic positives it let through (a right plan with one
  detail changed so the conversation contradicts it). Each names its file,
  `{where}/{CONTEXT_DIR}/dNN.md`: the conversation up to the reply, the reply, the judge's whole answer
  and the golden answer. Read every one before you write.

Past cycles:
{past}

## Write
One file, `{where}/changes.json`:

{CHANGES_SCHEMA}

`add`, `edit` and `remove` may each be empty, but together they must change something. Number edits
and removals as the champion's rubric numbers its lessons.

Rules for a lesson:
- It is general. It applies to any customer and any reservation. It never names a task, a reservation,
  flight or user id, a person, an amount from one conversation, or a disagreement id, and it never
  quotes a conversation. It may quote the policy.
- It is an instruction to the judge, one or two sentences, at most {LESSON_CHARS} characters. It says
  which check (1–5) it refines, or that it is about replies that are not plans.
- At most {MAX_ADDED} added this cycle, and at most {MAX_LESSONS} in the rubric.
- Prefer one lesson that fixes many disagreements to several that fix one each.
- Every lesson that makes the judge block more also risks blocking passes. Before you add one for the
  misses, check it against the false blocks: it must not turn any of them into a block.

Do not run anything. When you finish, the harness replays the new judge on every train checkpoint and
gates it on conversations you cannot see. Finish by writing `changes.json`.
"""


# ── the gate: paired by conversation, on the gate half ───────────────────────


def outcomes(scored: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each scored conversation balanced accuracy counts, and whether the judge got it right."""
    out: dict[str, dict[str, Any]] = {}
    for c in scored["conversations"]:
        if not c["scored"]:
            continue
        if c["passed"]:
            out[c["key"]] = {"half": c["half"], "kind": "pass", "right": c["first_block"] is None}
        elif c["wrong_plan"]:
            right = c["first_block"] == int(c["first_wrong"]["msg"])
            out[c["key"]] = {"half": c["half"], "kind": "wrong plan", "right": right}
    return out


def bars(scores: dict[str, Any]) -> dict[str, bool | None]:
    """Whether one half's scores meet each §9 bar; None where the half has nothing to measure."""
    out: dict[str, bool | None] = {}
    for key, (op, limit) in BARS.items():
        v = scores[key]
        x = v if key == "balanced_accuracy" else (v["k"] / v["n"] if v["n"] else None)
        out[key] = None if x is None else (x >= limit if op == ">=" else x <= limit)
    return out


def gate(
    domain: str,
    champion: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    gold: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The promotion decision, from both judges' verdicts on the gate half."""
    gold = gold_by_key(domain) if gold is None else gold
    a, b = replay.score(domain, champion, gold), replay.score(domain, challenger, gold)
    oa, ob = outcomes(a), outcomes(b)
    ka = {k for k, o in oa.items() if o["half"] == "gate"}
    kb = {k for k, o in ob.items() if o["half"] == "gate"}
    if ka != kb:
        raise ValueError(
            f"the replays score different gate conversations: {len(ka - kb)} only in the "
            f"champion's, {len(kb - ka)} only in the challenger's"
        )
    keys = sorted(ka)
    fixed = [k for k in keys if ob[k]["right"] and not oa[k]["right"]]
    broken = [k for k in keys if oa[k]["right"] and not ob[k]["right"]]
    p = mcnemar_one_sided(len(fixed), len(broken))
    ga, gb = a["scores"]["gate"], b["scores"]["gate"]
    ma, mb = bars(ga), bars(gb)
    lost = [k for k in BARS if ma[k] and mb[k] is False]
    ba_a, ba_b = ga["balanced_accuracy"], gb["balanced_accuracy"]
    rises = ba_a is not None and ba_b is not None and ba_b > ba_a
    significant = p < ALPHA
    dominant = len(fixed) >= MIN_FIXED and not broken
    promote = rises and (significant or dominant) and not lost
    rule = "mcnemar" if significant else "dominance" if dominant else "none"
    why = [
        f"balanced accuracy {ba_a} → {ba_b} ({'rises' if rises else 'does not rise'})",
        f"McNemar one-sided: fixed {len(fixed)}, broke {len(broken)}, p = {p:.3f} "
        f"{'<' if significant else '≥'} α = {ALPHA}",
    ]
    if dominant and not significant:
        why.append(f"dominance: broke 0, fixed ≥ {MIN_FIXED}")
    if lost:
        why.append(f"loses a bar the champion meets: {', '.join(lost)}")
    return {
        "promote": promote,
        "rule": rule,
        "reason": " · ".join(why),
        "conversations": len(keys),
        "fixed": [{"key": k, "kind": ob[k]["kind"]} for k in fixed],
        "broken": [{"key": k, "kind": oa[k]["kind"]} for k in broken],
        "p_value": round(p, 4),
        "balanced_accuracy": {"champion": ba_a, "challenger": ba_b},
        "bars": {k: {"champion": ma[k], "challenger": mb[k]} for k in BARS},
        "read": {
            "champion": a["scores"]["read"]["balanced_accuracy"],
            "challenger": b["scores"]["read"]["balanced_accuracy"],
        },
        "scores": {"champion": ga, "challenger": gb},
    }


def read_fixes(dis: list[dict[str, Any]], challenger: list[dict[str, Any]]) -> dict[str, int]:
    """Of the disagreements the optimiser read, how many the challenger now answers as gold does."""
    by = {(v["item"], v["key"], v["msg"]): v["verdict"] for v in challenger}
    fixed = sum(by.get((d["item"], d["key"], d["msg"])) == d["golden"]["verdict"] for d in dis)
    return {"read": len(dis), "now_agree": fixed}


# ── one cycle ────────────────────────────────────────────────────────────────


def _checksum(exclude: Path) -> dict[str, int]:
    """loop/optimiser.py's guarded tree, plus every judge, replay and judge data file."""
    out = tree_checksum(exclude)
    for rel in ("judges", "judge_runs", "data/judge"):
        for p in (ROOT / rel).rglob("*"):
            if p.is_file() and exclude not in p.parents:
                st = p.stat()
                out[str(p.relative_to(ROOT))] = int(st.st_mtime_ns) ^ st.st_size
    return out


def _yaml(champion: prompt.JudgeVersion, name: str, cycle: int) -> str:
    text = (champion.path / "judge.yaml").read_text()
    text = re.sub(
        r"\A#[^\n]*",
        f"# The plan judge {name} (s11 J3, cycle {cycle}): {champion.name}'s model, effort and decision "
        "rule, with the loop's lessons in judge.md beside it.",
        text,
        count=1,
    )
    return re.sub(r"(?m)^name: \S+", f"name: {name}\nparent: {champion.name}", text, count=1)


def _optimise(
    domain: str,
    champion: prompt.JudgeVersion,
    name: str,
    new_dir: Path,
    dis: list[dict[str, Any]],
    read_scores: dict[str, Any],
    history: list[dict[str, Any]],
    model: str,
    effort: Effort,
    log: Callable[[str], None],
) -> tuple[dict[str, Any], list[str]]:
    """The fenced optimiser session: it reads the bundle and writes `changes.json`, nothing else."""
    ctx = _write_context(domain, new_dir)
    write_bundle(ctx, champion, dis, read_scores)
    text = build_prompt(domain, champion, name, dis, history)
    before = _checksum(new_dir)
    transcript: list[dict[str, Any]] = []
    started = time.time()
    sess = asyncio.run(
        _run_session(
            text,
            label="judge optimiser",
            domain=domain,
            new_name=name,
            writable={(new_dir / "changes.json").resolve()},
            readable=set(),
            hidden=hidden_paths(domain),
            model=model,
            effort=effort,
            transcript=transcript,
            where=f"judges/{domain}/{KIND}/{name}/",
        )
    )
    optimiser: dict[str, Any] = {
        "model": model,
        "effort": effort,
        "turns": sess.turns,
        "input_tokens": sess.input_tokens,
        "output_tokens": sess.output_tokens,
        "duration_ms": int((time.time() - started) * 1000),
        "error": sess.error,
    }
    shutil.rmtree(new_dir / CONTEXT_DIR, ignore_errors=True)
    (new_dir / "optimiser_transcript.json").write_text(
        redact(json.dumps(transcript, ensure_ascii=False, indent=1))
    )
    minutes = optimiser["duration_ms"] / 60000
    log(
        f"  optimiser: {sess.turns} turns, {minutes:.1f} min"
        + (f", error {sess.error}" if sess.error else "")
    )
    after = _checksum(new_dir)
    touched = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    problems = (
        [f"the optimiser changed files outside {new_dir.relative_to(ROOT)}: {touched[:5]}"]
        if touched
        else []
    )
    if sess.error and not (new_dir / "changes.json").is_file():
        problems.append(f"the session ended with {sess.error}")
    return optimiser, problems


def run_cycle(
    domain: str = "airline",
    concurrency: int = 4,
    model: str = OPTIMISER_MODEL,
    effort: Effort = OPTIMISER_EFFORT,
    log: Callable[[str], None] = print,
    track: bool = True,
    recheck: str | None = None,
    why: str | None = None,
) -> dict[str, Any]:
    """One cycle: optimise the champion's lessons, replay the challenger, gate it, record it.

    `recheck` names a challenger whose cycle the harness rejected; its `changes.json` is checked
    again (after a fix to a guard, say `why`) and, if it passes, replayed and gated, with no new
    optimiser session. The rejected line stays in the ledger; the re-check is a line of its own,
    under the same cycle number."""
    require_live()
    reg = read_registry(domain)
    champion = prompt.load(domain, reg["champion"], KIND)
    champ_dir = replay.latest(domain, champion.name)
    if champ_dir is None:
        log(f"no full replay of {champion.name}: replaying it first")
        champ_dir = replay.run_replay(
            domain, champion.name, concurrency=concurrency, log=log, track=track
        )
    champ_meta = json.loads((champ_dir / "run.json").read_text())
    if champ_meta["fingerprint"] != champion.fingerprint:
        raise ValueError(
            f"{champion.name} changed since its replay {champ_dir.name}: replay it again first"
        )
    champ_v = replay.read_verdicts(champ_dir)
    history = read_ledger(domain)
    policy = str(read_task_extract(domain).get("policy") or "")
    dis = disagreements(domain, champ_v)
    ids = {d["id"]: f"{d['key']}#{d['msg']}" for d in dis}

    prev: dict[str, Any] | None = None
    if recheck:
        prev = next(
            (
                e
                for e in reversed(history)
                if e["challenger"]["name"] == recheck and e["verdict"] == "rejected"
            ),
            None,
        )
        if prev is None or prev["champion"]["name"] != champion.name:
            raise ValueError(
                f"{recheck} is not a rejected challenger of the champion {champion.name}"
            )
        if prev["read"]["ids"] != ids:
            raise ValueError("the read half's disagreements changed since; run a new cycle instead")
        cycle, name = int(prev["cycle"]), recheck
        new_dir = plan_dir(domain) / name
        optimiser: dict[str, Any] = {**prev["optimiser"], "reused": True}
        problems: list[str] = []
        log(f"cycle {cycle}: re-checking {name}'s changes.json ({why or 'no reason given'})")
    else:
        cycle = len({e["cycle"] for e in history}) + 1
        name = next_name(domain)
        new_dir = plan_dir(domain) / name
        new_dir.mkdir(parents=True)
        log(
            f"cycle {cycle}: {champion.name} → {name}; the optimiser reads {len(dis)} read-half disagreements"
        )
        read_scores = replay.score(domain, champ_v)["scores"]["read"]
        optimiser, problems = _optimise(
            domain, champion, name, new_dir, dis, read_scores, history, model, effort, log
        )

    changes_path = new_dir / "changes.json"
    changes: dict[str, Any] = {}
    if not changes_path.is_file():
        problems.append("no changes.json written")
    else:
        try:
            changes = json.loads(redact(changes_path.read_text()))
        except json.JSONDecodeError as e:
            problems.append(f"changes.json unreadable: {e}")
    head, lessons = split_rubric(champion.rubric)
    new_lessons, delta, errors = (
        apply_changes(lessons, changes)
        if changes
        else (lessons, {"added": [], "edited": [], "removed": []}, [])
    )
    problems += errors
    if changes:
        problems += lesson_problems(new_lessons, domain, by_conversation(dis), policy)

    entry: dict[str, Any] = {
        "cycle": cycle,
        "at": datetime.now(UTC).isoformat(),
        "domain": domain,
        "champion": {
            "name": champion.name,
            "fingerprint": champion.fingerprint,
            "replay": champ_dir.name,
        },
        "challenger": {"name": name, "fingerprint": None, "replay": None},
        "optimiser": optimiser,
        "read": {"disagreements": len(dis), "ids": ids},
        "diagnosis": changes.get("diagnosis"),
        "patterns": changes.get("patterns") or [],
        "expect": changes.get("expect"),
        "because": {
            k: [e.get("because") for e in changes.get(k) or [] if isinstance(e, dict)]
            for k in ("add", "edit", "remove")
        },
        "lessons": delta,
        "labels_sha": replay._sha(labels_path(domain)),
        "gold_sha": replay._sha(gold_path(domain)),
    }
    if prev is not None:
        entry["recheck"] = {"of": prev["at"], "was": prev["reason"], "why": why}
    if problems:
        entry.update(verdict="rejected", reason="; ".join(problems))
        log(f"  rejected: {entry['reason']}")
    else:
        (new_dir / "judge.md").write_text(render_rubric(head, new_lessons, name))
        (new_dir / "judge.yaml").write_text(_yaml(champion, name, cycle))
        challenger = prompt.load(domain, name, KIND)
        entry["challenger"]["fingerprint"] = challenger.fingerprint
        log(
            f"  {name}: {len(new_lessons)} lessons ({len(delta['added'])} added); "
            "replaying it on every train checkpoint"
        )
        chal_dir = replay.run_replay(domain, name, concurrency=concurrency, log=log, track=track)
        chal_v = replay.read_verdicts(chal_dir)
        entry["challenger"]["replay"] = chal_dir.name
        g = gate(domain, champ_v, chal_v)
        entry["gate"] = {k: v for k, v in g.items() if k != "scores"}
        entry["read"].update(read_fixes(dis, chal_v))
        entry.update(verdict="promoted" if g["promote"] else "held", reason=g["reason"])
        log(f"  {entry['verdict']}: {g['reason']}")
    redact_tree(new_dir)

    with ledger_path(domain).open("a") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    version = {
        "name": name,
        "made": f"judge loop, cycle {cycle}" + (" (re-checked)" if prev else ""),
        "cycle": cycle,
        "parent": champion.name,
        "verdict": entry["verdict"],
    }
    reg["versions"] = [v for v in reg["versions"] if v["name"] != name] + [version]
    if entry["verdict"] == "promoted":
        reg["champion"] = name
    registry_path(domain).write_text(json.dumps(reg, indent=1) + "\n")
    if track:
        from tau2_loop.tooljudge import tracking

        log(f"  MLflow: {tracking.log_cycle(entry)}")
    return entry


def last_verdicts(domain: str) -> list[str]:
    """Each cycle's verdict, its latest line (a re-check) standing for it."""
    by: dict[int, str] = {}
    for e in read_ledger(domain):
        by[int(e["cycle"])] = str(e["verdict"])
    return [by[k] for k in sorted(by)]


def run_loop(
    domain: str = "airline",
    cycles: int = 1,
    concurrency: int = 4,
    log: Callable[[str], None] = print,
    track: bool = True,
) -> list[dict[str, Any]]:
    """Up to `cycles` cycles; §7 stops after two holds in a row, or 5 cycles in all."""
    out = []
    for _ in range(cycles):
        if len(last_verdicts(domain)) >= 5:
            log("stopped: 5 cycles, §7's limit")
            break
        out.append(run_cycle(domain, concurrency=concurrency, log=log, track=track))
        if last_verdicts(domain)[-2:] == ["held", "held"]:
            log("stopped: two holds in a row")
            break
    return out
