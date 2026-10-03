"""J0 · labels: every checkpoint of every scored train conversation, from gold, with no model call.

The rule (s11, 2026-10-02): the judge learns from optimised answering agents only. A version the
loop produced (v1 onwards) is what the judge will guard; v0, the hand-written baseline, fails in
ways no optimised agent does (it tells a user it cannot see their payment methods and never calls
`get_user_details`), and a golden set built on it is a golden set about hallucinations. So v0's
runs never enter the labels, the golden answers, the replays or the judge loop.

The second rule (2026-10-02): the judge is called only when the agent issues a tool call that acts,
a write or a transfer, and before it runs; never on a text reply. Those checkpoints are `judged`.
The plan and reply checkpoints stay in the file, because golden answers are keyed by position
(`gold.checkpoint_ids`), and `live` still marks what the retired plan judge (j1–j3) reviewed.

A checkpoint is a moment the live judge could stop: a text reply (a plan, or another reply the
plan trigger flags), a write call, or a transfer. Each gets a label from the task's gold actions,
which no judge ever sees:

- a **write** is `good` when tau2's own rule (`Action.compare_with_tool_call`: the name, then the
  gold action's `compare_args` or every argument) matches it to a gold write, `bad` when it
  matches none, and `errored` when the tool refused it, so nothing changed;
- a **plan** is the agent's last text message before an executed write: the confirmation the
  airline policy makes it ask for. It is `right` when every write it led to was good, `wrong`
  when one was bad and the error was already in the plan, and `slip` when the plan was right and
  the call departed from it (a hand reading, `CALL_SLIPS`);
- a **reply** the trigger flags that is not a plan is `allow` in a passing conversation and
  `unlabelled` in a failed one, where J1's golden answers decide it;
- a **transfer** is `allow` in a pass and a `candidate` where a gold write never ran (J7's).

Tasks sit in the split they are in today (`data/splits/<domain>.json`), so test never mixes in:
its conversations are counted, never written. Train tasks are dealt into five folds by their
count of wrong plans and slips; J3's loop reads F1–F3 and is gated on F4–F5, so a task never sits
on both sides.

Synthetic positives are copies of right plans with one detail changed so the transcript
contradicts it, each kept in its source task's fold. This module reproduces every count on the
s11 plan page; `make judge-labels` writes `data/judge/<domain>.json`.
"""

from __future__ import annotations

import collections
import functools
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tau2_loop.config import DATA_DIR, RUNS_DIR
from tau2_loop.data.splits import read_split, read_task_extract

JUDGE_DATA_DIR = DATA_DIR / "judge"
LABELS_VERSION = 2

# The fallback trigger until a reply declares its phase (s11 Fig 4): it found 101 of the 102
# train plans, and fired on 416 of the 1,153 other text replies.
PLAN_TRIGGER = re.compile(r"(proceed|go ahead|confirm|\byes\b|correct\?|would you like)", re.I)
TRANSFER = "transfer_to_human_agents"
FOLDS = 5
HALVES: dict[str, tuple[int, ...]] = {"read": (1, 2, 3), "gate": (4, 5)}

#: The checkpoints the judge reviews, before the call runs: tool calls that change something.
JUDGED_KINDS: frozenset[str] = frozenset({"write", "transfer"})

#: Versions whose traces the judge never learns from: the baseline, before any loop cycle.
UNOPTIMISED: frozenset[str] = frozenset({"v0"})

#: The judge's data is closed (the person's call, 3 Oct 2026): a run id (its UTC start stamp) at or
#: after this never joins the labels, golden answers, replays, the judge loop, the pending queue
#: or Evals, whatever agent or domain it is. The last run the committed labels hold is 20260928.
CLOSED_AT = "20261002T000000Z"


def open_to_judge(run_id: str) -> bool:
    """Whether a run was scored before the judge's data closed."""
    return run_id < CLOSED_AT


# Folds pinned as J3's cycle 1 dealt them (2026-10-01): its optimiser read F1–F3, so re-dealing
# after v0 left could move a task it read into the gate half. A new domain is dealt by `task_folds`.
PINNED_FOLDS: dict[str, dict[str, int]] = {
    "airline": {
        t: f
        for f, ts in enumerate(
            [
                ["0", "4", "23", "38", "39"],
                ["6", "9", "30", "40", "47"],
                ["10", "22", "33", "35", "45"],
                ["11", "14", "20", "25", "46"],
                ["13", "19", "24", "44", "48"],
            ],
            start=1,
        )
        for t in ts
    },
}

# s08 §5: tasks with open problems on tau2 1.0.1 (simulator faults, grader bugs, policy
# loopholes). Metrics are reported with and without them.
SUSPECT: dict[str, frozenset[str]] = {
    "airline": frozenset({"35", "7", "6", "14", "2", "5", "13", "27", "29", "32", "45"}),
}

# A hand reading of each bad train write against the plan before it: these four followed a right
# plan, and the call departed from it. Keyed (run, task, the write's message index). Every other
# bad write carried out a plan that was already wrong when the user confirmed it.
CALL_SLIPS: dict[str, frozenset[tuple[str, str, int]]] = {
    "airline": frozenset(
        {
            ("20260928T094017Z_airline_v5_train", "14", 24),  # plan JFK→SFO, call destination JFK
            ("20260928T094017Z_airline_v5_train", "23", 26),  # plan "JFK to SFO", call JFK
            ("20260928T094017Z_airline_v5_train", "24", 32),  # plan JFK→SEA, call destination JFK
            (
                "20260928T094017Z_airline_v5_train",
                "33",
                26,
            ),  # plan named no payment; call chose one
        }
    ),
}


def labels_path(domain: str) -> Path:
    return JUDGE_DATA_DIR / f"{domain}.json"


def match(gold: dict[str, Any], name: str, args: dict[str, Any]) -> bool:
    """tau2's `Action.compare_with_tool_call`, on plain dicts."""
    if gold["name"] != name:
        return False
    keys = list(args) if gold.get("compare_args") is None else list(gold["compare_args"])
    if not keys:
        return True
    gold_args = gold.get("arguments") or {}
    return {k: v for k, v in args.items() if k in keys} == {
        k: v for k, v in gold_args.items() if k in keys
    }


def is_text_reply(m: dict[str, Any]) -> bool:
    return m.get("role") == "assistant" and bool(m.get("content")) and not m.get("tool_calls")


def fires(text: str) -> bool:
    return bool(PLAN_TRIGGER.search(text or ""))


def _errored(result: dict[str, Any]) -> bool:
    return bool(result.get("error")) or str(result.get("content") or "").startswith("Error")


def optimised(meta: dict[str, Any]) -> bool:
    """Whether a run's answering agent came out of the loop: the judge learns from no other."""
    return str(meta.get("agent") or "") not in UNOPTIMISED


def _scored_traces(domain: str) -> Iterator[tuple[str, dict[str, Any], str, dict[str, Any]]]:
    """(run id, run.json, trace file name, trace) for every finished, scored run of an optimised
    agent of the domain."""
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        meta_path, traces = run_dir / "run.json", run_dir / "traces"
        if not open_to_judge(run_dir.name) or not meta_path.is_file() or not traces.is_dir():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("domain") != domain or meta.get("dry_run") or not meta.get("finished_at"):
            continue
        if not optimised(meta):
            continue
        for f in sorted(traces.iterdir()):
            if f.suffix == ".json":
                yield run_dir.name, meta, f.name, json.loads(f.read_text())


def label_conversation(
    run: str,
    meta: dict[str, Any],
    trace_name: str,
    t: dict[str, Any],
    gold_actions: list[dict[str, Any]],
    kinds: dict[str, str],
    slips: frozenset[tuple[str, str, int]] = frozenset(),
) -> dict[str, Any]:
    """One conversation's checkpoints, in message order, each with its label."""
    task = str(t["task_id"])
    gold = [a for a in gold_actions if kinds.get(a["name"]) == "write"]
    reward = float((t.get("reward_info") or {}).get("reward") or 0)
    passed = reward >= 1.0
    msgs: list[dict[str, Any]] = t.get("messages") or []
    results = {m.get("id"): m for m in msgs if m.get("role") == "tool"}

    writes: list[dict[str, Any]] = []
    all_calls: list[tuple[int, dict[str, Any]]] = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            all_calls.append((i, tc))
            if kinds.get(tc["name"]) != "write":
                continue
            args = tc.get("arguments") or {}
            label = (
                "errored"
                if _errored(results.get(tc.get("id")) or {})
                else "good"
                if any(match(g, tc["name"], args) for g in gold)
                else "bad"
            )
            writes.append(
                {
                    "msg": i,
                    "call_id": tc.get("id"),
                    "name": tc["name"],
                    "arguments": args,
                    "label": label,
                }
            )

    # each executed write's plan: the agent's last text message before it
    plan_of: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for w in writes:
        if w["label"] == "errored":
            continue
        prev = [
            j
            for j, m in enumerate(msgs[: w["msg"]])
            if m.get("role") == "assistant" and m.get("content")
        ]
        if prev:
            plan_of[prev[-1]].append(w)

    def plan_label(ws: list[dict[str, Any]]) -> str:
        bad = [w for w in ws if w["label"] == "bad"]
        if not bad:
            return "right"
        return "slip" if all((run, task, w["msg"]) in slips for w in bad) else "wrong"

    ran = [(w["name"], w["arguments"]) for w in writes if w["label"] != "errored"]
    done = [g for g in gold if any(match(g, n, a) for n, a in ran)]
    transfer = any(tc["name"] == TRANSFER for _, tc in all_calls)
    if passed:
        mode = "pass"
    elif any(w["label"] == "bad" for w in writes):
        mode = "wrong_write"
    elif len(done) < len(gold):
        mode = "missed_write_transfer" if transfer else "missed_write_other"
    else:
        mode = "communicate"

    cps: list[dict[str, Any]] = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        if m.get("content") and i in plan_of:
            trig = fires(str(m["content"])) and not m.get("tool_calls")
            lab = plan_label(plan_of[i])
            cps.append(
                {
                    "msg": i,
                    "kind": "plan",
                    "judge": "plan",
                    "trigger": trig,
                    "live": trig,
                    "label": lab,
                    "pinned": lab == "right",
                    "writes": [w["msg"] for w in plan_of[i]],
                }
            )
        elif is_text_reply(m) and fires(str(m["content"])):
            cps.append(
                {
                    "msg": i,
                    "kind": "reply",
                    "judge": "plan",
                    "trigger": True,
                    "live": True,
                    "label": "allow" if passed else "unlabelled",
                    "pinned": passed,
                }
            )
        for w in (w for w in writes if w["msg"] == i):
            cps.append(
                {
                    "msg": i,
                    "kind": "write",
                    "judge": "call",
                    "trigger": False,
                    "live": False,
                    "label": w["label"],
                    "pinned": w["label"] in ("good", "bad"),
                    "name": w["name"],
                    "call_id": w["call_id"],
                    "arguments": w["arguments"],
                }
            )
        for tc in m.get("tool_calls") or []:
            if tc["name"] == TRANSFER:
                lab = (
                    "allow"
                    if passed
                    else "candidate"
                    if mode == "missed_write_transfer"
                    else "unlabelled"
                )
                cps.append(
                    {
                        "msg": i,
                        "kind": "transfer",
                        "judge": "direction",
                        "trigger": False,
                        "live": False,
                        "label": lab,
                        "pinned": passed,
                        "name": TRANSFER,
                        "call_id": tc.get("id"),
                    }
                )

    for c in cps:
        c["judged"] = c["kind"] in JUDGED_KINDS
    # the first wrong step among what a live judge sees in order: trigger-flagged replies and
    # writes (a plan the trigger missed is reached only at its write)
    seen = [c for c in cps if c["live"] or c["kind"] == "write"]
    first = next((c for c in seen if c["label"] in ("wrong", "bad")), None)
    trial = int(t.get("trial") or 1)
    return {
        "key": f"{run}/{task}/t{trial}",
        "run": run,
        "version": meta.get("agent"),
        "model": meta.get("model"),
        "task": task,
        "trial": trial,
        "trace": trace_name,
        "passed": passed,
        "reward": reward,
        "mode": mode,
        "transfer": transfer,
        "n_messages": len(msgs),
        "checkpoints": cps,
        "first_wrong": {"msg": first["msg"], "kind": first["kind"]} if first else None,
    }


def task_folds(convs: list[dict[str, Any]], train: list[str]) -> dict[str, int]:
    """Each train task's fold, 1–5: tasks with the most wrong plans and slips first, dealt in snake
    order so every fold holds a share of the positives."""
    per_task: collections.Counter[str] = collections.Counter(
        c["task"]
        for c in convs
        for cp in c["checkpoints"]
        if cp["kind"] == "plan" and cp["label"] != "right"
    )
    order = sorted(train, key=lambda t: (-per_task[t], int(t) if t.isdigit() else t))
    fold_of: dict[str, int] = {}
    for i, t in enumerate(order):
        r, k = divmod(i, FOLDS)
        fold_of[t] = (k if r % 2 == 0 else FOLDS - 1 - k) + 1
    return fold_of


def half_of(fold: int | None) -> str | None:
    return next((h for h, fs in HALVES.items() if fold in fs), None)


# ── synthetic positives ──────────────────────────────────────────────────────

_ARROW = r"\s*(?:to|→|->|–|—|-)\s*"


def _perturb_payment(text: str, args: dict[str, Any]) -> tuple[str, dict[str, str]] | None:
    ids: list[str] = []
    if isinstance(args.get("payment_id"), str):
        ids.append(args["payment_id"])
    for p in args.get("payment_methods") or []:
        if isinstance(p, dict) and isinstance(p.get("payment_id"), str):
            ids.append(p["payment_id"])
    for pid in ids:
        if pid in text and "_" in pid:
            fake = f"{pid.rsplit('_', 1)[0]}_{'9' * 7}"
            return text.replace(pid, fake), {"from": pid, "to": fake}
    return None


def _perturb_route(text: str, args: dict[str, Any]) -> tuple[str, dict[str, str]] | None:
    o, d = args.get("origin"), args.get("destination")
    if not (isinstance(o, str) and isinstance(d, str)) or o == d:
        return None
    m = re.search(rf"\b{re.escape(o)}({_ARROW}){re.escape(d)}\b", text)
    if not m:
        return None
    new = f"{o}{m.group(1)}{o}"
    return text[: m.start()] + new + text[m.end() :], {"from": m.group(0), "to": new}


def _perturb_passenger(text: str, args: dict[str, Any]) -> tuple[str, dict[str, str]] | None:
    """A booking's last passenger dropped from the plan: a list line, or an item of a comma list."""
    people = [p for p in args.get("passengers") or [] if isinstance(p, dict)]
    if (
        len(people) < 2 or "flights" not in args
    ):  # book_reservation only: a name change is not a booking
        return None
    first, last = (
        f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        for p in (people[0], people[-1])
    )
    if not last or first not in text:
        return None
    name = re.escape(last)
    for pat in (
        rf"(?m)^[ \t]*(?:[-*•]|\d+\.)[ \t]*(?:\*\*)?{name}\b.*(?:\n|$)",
        rf",[ \t]*(?:and[ \t]+)?{name}\b(?:[ \t]*\([^)\n]*\))?",
        rf"[ \t]+and[ \t]+{name}\b(?:[ \t]*\([^)\n]*\))?",
    ):
        m = re.search(pat, text)
        if m:
            return text[: m.start()] + text[m.end() :], {"from": m.group(0).strip(), "to": ""}
    return None


PERTURB = {
    "payment": ("a payment id that is not on the user's profile", _perturb_payment),
    "route": ("the destination set to the origin", _perturb_route),
    "passenger": ("one passenger short", _perturb_passenger),
}


def synthetic_positives(
    convs: list[dict[str, Any]], msgs_of: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """One perturbed copy of each right plan that admits one, the scarcest kind first.

    The change is to the plan text only; the transcript before it is the real one, so the
    contradiction is checkable: a payment id the profile never returned, a route the listed
    flights don't fly, a passenger the user named and the plan dropped."""
    out: list[dict[str, Any]] = []
    made: collections.Counter[str] = collections.Counter()
    for c in convs:
        msgs = msgs_of[c["key"]]
        for cp in c["checkpoints"]:
            if cp["kind"] != "plan" or cp["label"] != "right" or not cp["live"]:
                continue
            text = str(msgs[cp["msg"]].get("content") or "")
            later = [
                x for x in c["checkpoints"] if x["kind"] == "write" and x["msg"] in cp["writes"]
            ]
            options: list[tuple[str, str, dict[str, str]]] = []
            for kind, (_, fn) in PERTURB.items():
                for w in later:
                    got = fn(text, w.get("arguments") or {})
                    if got and got[0] != text:
                        options.append((kind, got[0], got[1]))
                        break
            if not options:
                continue
            kind, new, change = min(options, key=lambda o: made[o[0]])
            made[kind] += 1
            out.append(
                {
                    "id": f"S{len(out) + 1:03d}",
                    "key": c["key"],
                    "msg": cp["msg"],
                    "task": c["task"],
                    "fold": c["fold"],
                    "half": c["half"],
                    "kind": kind,
                    "what": PERTURB[kind][0],
                    "change": change,
                    "text": new,
                    "label": "wrong",
                    "check": 2,
                }
            )
    return out


# ── the label file ───────────────────────────────────────────────────────────


def build(domain: str = "airline") -> dict[str, Any]:
    """Label every scored conversation of the domain. Train is written whole; test is counted."""
    split = read_split(domain)
    train, test = set(split["train"]), set(split["test"])
    ext = read_task_extract(domain)
    kinds = {t["name"]: str(t.get("type")) for t in ext.get("tools", [])}
    tasks = {str(t["id"]): t for t in ext.get("tasks", [])}
    slips = CALL_SLIPS.get(domain, frozenset())
    suspect = SUSPECT.get(domain, frozenset())

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    msgs_of: dict[str, list[dict[str, Any]]] = {}
    for run, meta, name, t in _scored_traces(domain):
        tid = str(t["task_id"])
        side = "train" if tid in train else "test" if tid in test else None
        if side is None or tid not in tasks:
            continue
        gold_actions = (tasks[tid].get("evaluation_criteria") or {}).get("actions") or []
        c = label_conversation(run, meta, name, t, gold_actions, kinds, slips)
        c["split"], c["suspect"] = side, tid in suspect
        by_split[side].append(c)
        if side == "train":
            msgs_of[c["key"]] = t.get("messages") or []

    convs = by_split["train"]
    fold_of = PINNED_FOLDS.get(domain) or task_folds(convs, sorted(train))
    if set(fold_of) != train:
        raise ValueError(f"{domain}'s pinned folds no longer match its train split: re-deal them")
    for c in convs:
        c["fold"] = fold_of.get(c["task"])
        c["half"] = half_of(c["fold"])
    synth = synthetic_positives(convs, msgs_of)
    return {
        "version": LABELS_VERSION,
        "domain": domain,
        "split_version": split.get("version"),
        "made_by": "make judge-labels (src/tau2_loop/tooljudge/labels.py), no model call",
        "trigger": PLAN_TRIGGER.pattern,
        "folds": {
            f"F{f}": sorted(
                (t for t, k in fold_of.items() if k == f),
                key=lambda t: int(t) if t.isdigit() else 0,
            )
            for f in range(1, FOLDS + 1)
        },
        "halves": {h: [f"F{f}" for f in fs] for h, fs in HALVES.items()},
        "summary": {
            "train": summarise(convs, msgs_of),
            "test": summarise(by_split["test"], None),
            "synthetic": dict(collections.Counter(s["kind"] for s in synth)),
            "by_fold": {
                f"F{f}": summarise([c for c in convs if c["fold"] == f], None)
                for f in range(1, FOLDS + 1)
            },
            "by_half": {h: summarise([c for c in convs if c["half"] == h], None) for h in HALVES},
        },
        "conversations": convs,
        "synthetic": synth,
    }


def summarise(
    convs: list[dict[str, Any]], msgs_of: dict[str, list[dict[str, Any]]] | None
) -> dict[str, Any]:
    """The counts the plan page quotes, from labelled conversations."""
    cps = [cp for c in convs for cp in c["checkpoints"]]
    out: dict[str, Any] = {
        "conversations": len(convs),
        "passed": sum(c["passed"] for c in convs),
        "failed": sum(not c["passed"] for c in convs),
        "modes": dict(collections.Counter(c["mode"] for c in convs)),
        "plans": dict(collections.Counter(cp["label"] for cp in cps if cp["kind"] == "plan")),
        "writes": dict(collections.Counter(cp["label"] for cp in cps if cp["kind"] == "write")),
        "transfers": {
            "outputs": sum(cp["kind"] == "transfer" for cp in cps),
            "conversations": sum(c["transfer"] for c in convs),
            "in_passes": sum(c["transfer"] and c["passed"] for c in convs),
            "in_missed_write_failures": sum(c["mode"] == "missed_write_transfer" for c in convs),
        },
        "trigger_fires": {
            "plans": sum(cp["kind"] == "plan" and cp["trigger"] for cp in cps),
            "plans_missed": sum(cp["kind"] == "plan" and not cp["trigger"] for cp in cps),
            "replies": sum(cp["kind"] == "reply" for cp in cps),
        },
        "first_wrong": dict(
            collections.Counter(c["first_wrong"]["kind"] for c in convs if c["first_wrong"])
        ),
        "pinned": sum(cp["pinned"] for cp in cps),
        # what the judge reviews (the second rule): checkpoints, and the messages that hold them
        "judged": {
            "checkpoints": sum(cp["judged"] for cp in cps),
            "messages": sum(
                len({cp["msg"] for cp in c["checkpoints"] if cp["judged"]}) for c in convs
            ),
            "conversations": sum(any(cp["judged"] for cp in c["checkpoints"]) for c in convs),
        },
    }
    if msgs_of is not None:
        text_replies = sum(sum(1 for m in msgs_of[c["key"]] if is_text_reply(m)) for c in convs)
        out["trigger_fires"]["other_text_replies"] = (
            text_replies - out["trigger_fires"]["plans"] - out["trigger_fires"]["plans_missed"]
        )
    return out


def write(domain: str = "airline") -> tuple[Path, dict[str, Any]]:
    data = build(domain)
    p = labels_path(domain)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return p, data


@functools.lru_cache(maxsize=16)
def _read_json(path: Path, mtime_ns: int) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def read_json(path: Path) -> dict[str, Any] | None:
    """A committed JSON file, parsed once per modification (the viewer asks per request)."""
    return _read_json(path, path.stat().st_mtime_ns) if path.is_file() else None


def read_labels(domain: str) -> dict[str, Any] | None:
    return read_json(labels_path(domain))


def unlabelled_runs(domain: str) -> list[dict[str, Any]]:
    """Finished runs of an optimised agent with train traces that `make judge-labels` has not read
    yet: what a new experiment adds to the judge's eval set. Reads run.json and file names only."""
    data = read_labels(domain)
    have = {c["run"] for c in (data or {}).get("conversations") or []}
    train = set(read_split(domain)["train"])
    out = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        meta_path, traces = run_dir / "run.json", run_dir / "traces"
        if run_dir.name in have or not open_to_judge(run_dir.name):
            continue
        if not meta_path.is_file() or not traces.is_dir():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("domain") != domain or meta.get("dry_run") or not meta.get("finished_at"):
            continue
        n = sum(f.suffix == ".json" and f.stem.split("_")[0] in train for f in traces.iterdir())
        if optimised(meta) and n:
            out.append({"run": run_dir.name, "agent": meta.get("agent"), "traces": n})
    return out


def conversation_labels(domain: str, run: str, task: str, trial: int) -> dict[str, Any] | None:
    """One conversation's labelled record, or None when it is not a labelled train conversation."""
    data = read_labels(domain)
    if not data:
        return None
    key = f"{run}/{task}/t{trial}"
    return next((c for c in data["conversations"] if c["key"] == key), None)
