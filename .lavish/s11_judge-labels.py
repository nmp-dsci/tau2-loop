"""s11: label every write call and transfer in the committed airline runs, for the tool-judge plan.

No model call. A write is `good` when tau2's own rule (`Action.compare_with_tool_call`: name,
then the gold action's compare_args or every argument) matches it to one of the task's gold write
actions, `bad` when it matches none, and `errored` when the tool itself refused it (no state
change). Each conversation gets one failure mode: `wrong_write` (a bad write ran),
`missed_write` (a gold write never happened), `communicate`, or `pass`. Tasks sit in the split
they are in today (`data/splits/airline.json`, v2), so train and test never mix.

    uv run python .lavish/s11_judge-labels.py            # the counts on the s11 page
    uv run python .lavish/s11_judge-labels.py --bad      # the bad writes, one per line
    uv run python .lavish/s11_judge-labels.py --plans    # writes grouped by the plan before them
    uv run python .lavish/s11_judge-labels.py --folds    # train balance, intervals, task folds
"""

from __future__ import annotations

import collections
import re
import statistics
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = json.loads((ROOT / "data/tasks/airline.json").read_text())
SPLIT = json.loads((ROOT / "data/splits/airline.json").read_text())
KINDS = {t["name"]: t.get("type") for t in EXTRACT["tools"]}
TASKS = {t["id"]: t for t in EXTRACT["tasks"]}
TRAIN, TEST = set(SPLIT["train"]), set(SPLIT["test"])
# s08 §5: open problems on tau2 1.0.1 (simulator faults, grader bugs, policy loopholes)
SUSPECT = {"35", "7", "6", "14", "2", "5", "13", "27", "29", "32", "45"}
# A hand reading of each bad train write against the agent's last text message before it (its
# stated plan, which the policy makes it confirm with the user): `call` when that plan was right
# and the call departed from it, else the plan itself was wrong. Keyed (run, task, write message).
CALL_SLIPS = {
    ("20260928T094017Z_airline_v5_train", "14", 24),  # plan JFK→SFO, call destination JFK
    ("20260928T094017Z_airline_v5_train", "23", 26),  # plan "round trip JFK to SFO", call destination JFK
    ("20260928T094017Z_airline_v5_train", "24", 32),  # plan JFK→SEA, call destination JFK
    ("20260928T094017Z_airline_v5_train", "33", 26),  # plan named no payment; call picked another gift card
}


def match(gold: dict, name: str, args: dict) -> bool:
    """tau2's Action.compare_with_tool_call, on plain dicts."""
    if gold["name"] != name:
        return False
    keys = list(args) if gold.get("compare_args") is None else gold["compare_args"]
    if not keys:
        return True
    pick = lambda d: {k: v for k, v in d.items() if k in keys}  # noqa: E731
    return pick(args) == pick(gold["arguments"])


def label() -> tuple[list[dict], list[dict]]:
    writes, convs = [], []
    for run in sorted(p.name for p in (ROOT / "runs").iterdir() if "_airline_" in p.name):
        meta = json.loads((ROOT / "runs" / run / "run.json").read_text())
        traces = ROOT / "runs" / run / "traces"
        if meta.get("dry_run") or not meta.get("finished_at") or not traces.is_dir():
            continue
        for f in sorted(traces.iterdir()):
            t = json.loads(f.read_text())
            tid = str(t["task_id"])
            gold = [
                a
                for a in (TASKS[tid]["evaluation_criteria"].get("actions") or [])
                if KINDS.get(a["name"]) == "write"
            ]
            ri = t.get("reward_info") or {}
            passed = float(ri.get("reward") or 0) >= 1.0
            msgs = t["messages"]
            results = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
            calls = [
                (i, tc)
                for i, m in enumerate(msgs)
                if m.get("role") == "assistant"
                for tc in (m.get("tool_calls") or [])
            ]
            split = "train" if tid in TRAIN else "test" if tid in TEST else "other"
            mine = []
            for i, tc in calls:
                if KINDS.get(tc["name"]) != "write":
                    continue
                r = results.get(tc["id"]) or {}
                errored = bool(r.get("error")) or str(r.get("content") or "").startswith("Error")
                good = any(match(g, tc["name"], tc.get("arguments") or {}) for g in gold)
                mine.append(
                    dict(run=run, version=meta["agent"], split=split, task=tid, msg=i,
                         name=tc["name"], label="errored" if errored else "good" if good else "bad",
                         passed=passed, suspect=tid in SUSPECT, arguments=tc.get("arguments"))
                )
            done = [g for g in gold if any(match(g, tc["name"], tc.get("arguments") or {}) for _, tc in calls)]
            transfer = any(tc["name"] == "transfer_to_human_agents" for _, tc in calls)
            if passed:
                mode = "pass"
            elif any(w["label"] == "bad" for w in mine):
                mode = "wrong_write"
            elif len(done) < len(gold):
                mode = "missed_write_transfer" if transfer else "missed_write_other"
            else:
                mode = "communicate"
            convs.append(dict(run=run, version=meta["agent"], split=split, task=tid, mode=mode,
                              transfer=transfer, passed=passed, n_writes=len(mine)))
            writes.extend(mine)
    return writes, convs


PLAN_TRIGGER = re.compile(r"(proceed|go ahead|confirm|\byes\b|correct\?|would you like)", re.I)


def plans(writes: list[dict], convs: list[dict]) -> None:
    """Executed writes grouped by the agent's last text message before each (its stated plan),
    and how well a code trigger finds those messages among all the agent's text replies."""
    for split in ("train", "test"):
        groups: dict[tuple, list[dict]] = collections.defaultdict(list)
        for w in writes:
            if w["split"] != split or w["label"] == "errored":
                continue
            msgs = json.loads((ROOT / "runs" / w["run"] / "traces" / f"{w['task']}.json").read_text())["messages"]
            prev = [i for i, m in enumerate(msgs[: w["msg"]]) if m.get("role") == "assistant" and m.get("content")]
            groups[(w["run"], w["task"], prev[-1])].append(w)
        bad = [g for g in groups.values() if any(w["label"] == "bad" for w in g)]
        slips = [g for g in bad if all((w["run"], w["task"], w["msg"]) in CALL_SLIPS for w in g if w["label"] == "bad")]
        print(f"== {split}: {sum(len(g) for g in groups.values())} executed writes after {len(groups)} plans; "
              f"{sum(len(g) > 1 for g in groups.values())} plans cover 2+ writes; "
              f"{len(groups) - len(bad)} plans led only to good writes, {len(bad)} to a bad one"
              + (f" ({len(bad) - len(slips)} wrong plans, {len(slips)} right plans whose call slipped)" if split == "train" else ""))
        if split != "train":
            continue
        tp = fn = fp = tn = 0
        for c in (c for c in convs if c["split"] == split):
            msgs = json.loads((ROOT / "runs" / c["run"] / "traces" / f"{c['task']}.json").read_text())["messages"]
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant" or not m.get("content") or m.get("tool_calls"):
                    continue
                is_plan, hit = (c["run"], c["task"], i) in groups, bool(PLAN_TRIGGER.search(m["content"]))
                tp, fn, fp, tn = tp + (is_plan and hit), fn + (is_plan and not hit), fp + (hit and not is_plan), tn + (not hit and not is_plan)
        print(f"  plan trigger: {tp} of {tp + fn} plans, and {fp} of {fp + tn} other text replies")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


FOLDS = 5


def task_folds(writes: list[dict], convs: list[dict]) -> dict[str, int]:
    """Each train task's fold: tasks with the most wrong plans and slips first, dealt in snake order."""
    pos: collections.Counter = collections.Counter()
    for w in writes:
        if w["split"] == "train" and w["label"] == "bad":
            msgs = json.loads((ROOT / "runs" / w["run"] / "traces" / f"{w['task']}.json").read_text())["messages"]
            prev = [i for i, m in enumerate(msgs[: w["msg"]]) if m.get("role") == "assistant" and m.get("content")]
            pos[(w["run"], w["task"], prev[-1])] = 1
    per_task: collections.Counter = collections.Counter(key[1] for key in pos)
    tasks = sorted(SPLIT["train"], key=lambda t: (-per_task[t], int(t)))
    fold_of = {}
    for i, t in enumerate(tasks):
        r, k = divmod(i, FOLDS)
        fold_of[t] = k if r % 2 == 0 else FOLDS - 1 - k
    return fold_of


def folds(writes: list[dict], convs: list[dict]) -> None:
    """Train's balance, the precision each bar can reach on train alone, and five folds grouped by
    task (a task recurs across 8-10 runs, so a split by version would put it on both sides)."""
    cs = [c for c in convs if c["split"] == "train"]
    ws = [w for w in writes if w["split"] == "train" and w["label"] != "errored"]
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for w in ws:
        msgs = json.loads((ROOT / "runs" / w["run"] / "traces" / f"{w['task']}.json").read_text())["messages"]
        prev = [i for i, m in enumerate(msgs[: w["msg"]]) if m.get("role") == "assistant" and m.get("content")]
        groups[(w["run"], w["task"], prev[-1])].append(w)
    plan_label = {}
    for key, g in groups.items():
        bad = [w for w in g if w["label"] == "bad"]
        plan_label[key] = "right" if not bad else "slip" if all((w["run"], w["task"], w["msg"]) in CALL_SLIPS for w in bad) else "wrong"
    print(f"== train balance: {sum(c['passed'] for c in cs)} passed, {sum(not c['passed'] for c in cs)} failed of {len(cs)}")
    print(f"  plans: {collections.Counter(plan_label.values())}; writes: {collections.Counter(w['label'] for w in ws)}")
    print("  95% intervals at the bars:")
    for name, k, n in (("plan catch 8/15", 8, 15), ("plan false blocks 4/87", 4, 87),
                       ("write catch 10/20", 10, 20), ("write false blocks 6/117", 6, 117)):
        lo, hi = wilson(k, n)
        print(f"    {name}: {k / n:.1%} [{lo:.1%}, {hi:.1%}]")
    fold_of = task_folds(writes, convs)
    tasks = sorted(SPLIT["train"], key=int)
    print("  folds (tasks · conversations passed/failed · plans right/wrong/slip · writes good/bad):")
    for f in range(FOLDS):
        ts = sorted((t for t in tasks if fold_of[t] == f), key=int)
        fc = [c for c in cs if fold_of[c["task"]] == f]
        fp = collections.Counter(lab for key, lab in plan_label.items() if fold_of[key[1]] == f)
        fw = collections.Counter(w["label"] for w in ws if fold_of[w["task"]] == f)
        print(f"    F{f + 1}: tasks {','.join(ts)} · {sum(c['passed'] for c in fc)}/{sum(not c['passed'] for c in fc)}"
              f" · {fp['right']}/{fp['wrong']}/{fp['slip']} · {fw['good']}/{fw['bad']}")
    # the judge loop reads F1-F3 and is gated on F4-F5
    for name, fs in (("read half F1-F3", {0, 1, 2}), ("gate half F4-F5", {3, 4})):
        hc = [c for c in cs if fold_of[c["task"]] in fs]
        hp = collections.Counter(lab for key, lab in plan_label.items() if fold_of[key[1]] in fs)
        hw = collections.Counter(w["label"] for w in ws if fold_of[w["task"]] in fs)
        print(f"  {name}: conversations {sum(c['passed'] for c in hc)}/{sum(not c['passed'] for c in hc)} · "
              f"plans {hp['right']}/{hp['wrong']}/{hp['slip']} · writes {hw['good']}/{hw['bad']}")
    neg = sum(1 for lab in plan_label.values() if lab != "wrong") + sum(w["label"] == "good" for w in ws)
    pos = sum(1 for lab in plan_label.values() if lab == "wrong") + sum(w["label"] == "bad" for w in ws)
    print(f"  allow-everything judge: right on {neg} of {neg + pos} plans and writes ({neg / (neg + pos):.1%}), catching 0 of {pos}")
    print("  95% intervals at the gate-half bars:")
    for name, k, n in (("plan catch 3/5", 3, 5), ("plan false blocks 2/41", 2, 41), ("write false blocks 2/50", 2, 50),
                       ("passing conversations interrupted 2/55", 2, 55), ("wrong-write conversations caught in time 4/7", 4, 7),
                       ("wrong plans stopped at the plan 2/4", 2, 4)):
        lo, hi = wilson(k, n)
        print(f"    {name}: {k / n:.1%} [{lo:.1%}, {hi:.1%}]")


def frequency(writes: list[dict], convs: list[dict]) -> None:
    """Every checkpoint the live judge would see, in order, per train conversation: each text
    reply that fires the plan trigger, and each reply with write calls (errored ones included,
    since the judge sees a call before tau2 runs it). In a failed conversation, the first wrong
    checkpoint is the first wrong plan or bad write; what follows it is a transcript the live
    judge reaches only if it missed that step."""
    by_conv = collections.defaultdict(list)
    for w in writes:
        by_conv[(w["run"], w["task"])].append(w)
    tot = collections.Counter()
    per = {"passed": [], "wrong_write": [], "missed_write": [], "communicate": []}
    firsts: dict[tuple, str] = {}
    for c in (c for c in convs if c["split"] == "train"):
        msgs = json.loads((ROOT / "runs" / c["run"] / "traces" / f"{c['task']}.json").read_text())["messages"]
        ws = sorted(by_conv[(c["run"], c["task"])], key=lambda w: w["msg"])
        plan_of = {}
        for w in ws:
            if w["label"] == "errored":
                continue
            prev = [i for i, m in enumerate(msgs[: w["msg"]]) if m.get("role") == "assistant" and m.get("content")]
            plan_of.setdefault(prev[-1], []).append(w)
        cps = []  # (message index, kind, label)
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            if m.get("content") and not m.get("tool_calls") and PLAN_TRIGGER.search(m["content"]):
                if i in plan_of:
                    bad = [w for w in plan_of[i] if w["label"] == "bad"]
                    lab = "right" if not bad else "slip" if all((w["run"], w["task"], w["msg"]) in CALL_SLIPS for w in bad) else "wrong"
                    cps.append((i, "plan", lab))
                else:
                    cps.append((i, "reply", "allow" if c["passed"] else "unlabelled"))
            for w in (w for w in ws if w["msg"] == i):
                cps.append((i, "write", w["label"]))
        first = next((k for k, cp in enumerate(cps) if cp[2] in ("wrong", "bad")), None)
        if first is not None:
            firsts[(c["run"], c["task"])] = cps[first][1]
        group = "passed" if c["passed"] else "wrong_write" if c["mode"] == "wrong_write" else "missed_write" if c["mode"].startswith("missed") else "communicate"
        per[group].append(len(cps))
        for k, cp in enumerate(cps):
            where = "before" if first is None or k < first else "first" if k == first else "after"
            tot[(group, where, cp[1], cp[2])] += 1
    for g, ns in per.items():
        if ns:
            print(f"== {g}: {len(ns)} conversations, {sum(ns)} checkpoints ({sum(ns) / len(ns):.2f} each)")
            rows = collections.Counter()
            for (gg, where, kind, lab), n in tot.items():
                if gg == g:
                    rows[(where, kind, lab)] += n
            for key in sorted(rows):
                print(f"   {key}: {rows[key]}")
    # every agent output, by what the judge does with it
    kinds_out: collections.Counter = collections.Counter()
    for c in (c for c in convs if c["split"] == "train"):
        for m in json.loads((ROOT / "runs" / c["run"] / "traces" / f"{c['task']}.json").read_text())["messages"]:
            if m.get("role") != "assistant":
                continue
            calls = m.get("tool_calls") or []
            names = [tc["name"] for tc in calls]
            if any(KINDS.get(n) == "write" for n in names):
                kinds_out["reply with write calls · write check"] += 1
            elif "transfer_to_human_agents" in names:
                kinds_out["transfer · J7"] += 1
            elif calls:
                kinds_out["read or generic calls only · not judged"] += 1
            elif m.get("content") and PLAN_TRIGGER.search(m["content"]):
                kinds_out["text reply, trigger fires · plan check"] += 1
            else:
                kinds_out["text reply, no trigger · not judged"] += 1
    n_out, n_c = sum(kinds_out.values()), sum(1 for c in convs if c["split"] == "train")
    print(f"== agent outputs on train: {n_out} in {n_c} conversations ({n_out / n_c:.2f} each)")
    for k_, v in kinds_out.most_common():
        print(f"   {k_}: {v} ({v / n_c:.2f} a conversation, {v / n_out:.1%})")
    # reads: how many per conversation, and how many match one of the task's gold reads
    rd: collections.Counter = collections.Counter()
    for c in (c for c in convs if c["split"] == "train"):
        gold = [g for g in (TASKS[c["task"]]["evaluation_criteria"].get("actions") or []) if KINDS.get(g["name"]) == "read"]
        grp = "passed" if c["passed"] else "failed"
        rd[(grp, "conversations")] += 1
        for m in json.loads((ROOT / "runs" / c["run"] / "traces" / f"{c['task']}.json").read_text())["messages"]:
            for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
                if KINDS.get(tc["name"]) != "read":
                    continue
                rd[(grp, "reads")] += 1
                rd[(grp, "on gold")] += any(match(g, tc["name"], tc.get("arguments") or {}) for g in gold)
    for grp in ("passed", "failed"):
        n, r, on = rd[(grp, "conversations")], rd[(grp, "reads")], rd[(grp, "on gold")]
        print(f"== reads in {grp} conversations: {r} ({r / n:.2f} each); {on} match a gold read, {r - on} don't ({(r - on) / r:.0%})")
    # what a per-checkpoint false-block rate f does to passing conversations, and the net effect
    k = sum(per["passed"]) / len(per["passed"])
    n_pass, n_wrong = len(per["passed"]), len(per["wrong_write"])
    print(f"== compounding over {k:.2f} checkpoints per passing conversation (catch 1/2, heal 1/2, drift 1/4):")
    for f in (0.05, 0.02, 0.01):
        p_int = 1 - (1 - f) ** k
        lost, saved = n_pass * p_int * 0.25, n_wrong * 0.5 * 0.5
        print(f"   f={f:.0%}: {p_int:.1%} of {n_pass} passing interrupted ({n_pass * p_int:.1f}); lost {lost:.1f}, "
              f"recovered {saved:.2f} of {n_wrong}; net {saved - lost:+.1f}")
    # with a plan flag in the reply contract, only declared plans and writes are checkpoints
    flagged = sum(n for (g, _w, kind, _l), n in tot.items() if g == "passed" and kind in ("plan", "write"))
    print(f"== with a plan flag: {flagged} checkpoints in {n_pass} passing conversations ({flagged / n_pass:.2f} each, "
          f"against {k:.2f} with the regex trigger)")
    for f in (0.05, 0.02, 0.01):
        print(f"   f={f:.0%}: {1 - (1 - f) ** (flagged / n_pass):.1%} of passing conversations interrupted")
    # the plan judge alone: its checkpoints in a passing conversation, and what it can reach
    plan_only = {"regex": sum(n for (g, _w, kind, _l), n in tot.items() if g == "passed" and kind in ("plan", "reply")),
                 "declared": sum(n for (g, _w, kind, _l), n in tot.items() if g == "passed" and kind == "plan")}
    first_plan = sum(n for (g, w_, kind, _l), n in tot.items() if g == "wrong_write" and w_ == "first" and kind == "plan")
    print(f"== plan judge alone: reaches {first_plan} of {n_wrong} wrong-write conversations at their first wrong step")
    for trig, n_cp in plan_only.items():
        kk = n_cp / n_pass
        for f in (0.05, 0.02, 0.01):
            p_int = 1 - (1 - f) ** kk
            lost, saved = n_pass * p_int * 0.25, first_plan * 0.5 * 0.5
            print(f"   {trig}: {kk:.2f} checkpoints per pass; f={f:.0%}: {p_int:.1%} interrupted ({n_pass * p_int:.1f}), "
                  f"lost {lost:.1f}, recovered {saved:.2f}, net {saved - lost:+.1f}")
    # balanced accuracy weighs the two classes equally, so a stopped failure outweighs an interrupted pass
    print(f"== balanced accuracy: one more stopped wrong-plan conversation is worth {n_pass / first_plan:.1f} "
          f"interrupted passes on all train ({n_pass} vs {first_plan}); an allow-everything judge scores 0.50")
    fold_of = task_folds(writes, convs)
    for name, fs in (("read half", {0, 1, 2}), ("gate half", {3, 4})):
        hc = [c for c in convs if c["split"] == "train" and fold_of[c["task"]] in fs]
        print(f"== {name}: {sum(c['passed'] for c in hc)} passing, {sum(c['mode'] == 'wrong_write' for c in hc)} wrong-write, "
              f"{sum(c['mode'].startswith('missed') for c in hc)} missed-write conversations")
        fw = collections.Counter(firsts[(c["run"], c["task"])] for c in hc if (c["run"], c["task"]) in firsts)
        print(f"   first wrong step in its wrong-write conversations: {dict(fw)}")


def models() -> None:
    """The airline agent on the same 366-character stock prompt: Haiku 4.5 (v0) against Sonnet 5 (v3)
    and Opus 5.5 (v5), on the 20 train tasks split v1 and v2 share."""
    shared = set(SPLIT["v1"]["train"])
    for run in ("20260915T022857Z_airline_v0_train", "20260915T075151Z_airline_v0_train",
                "20260928T060029Z_airline_v3_train", "20260928T093954Z_airline_v3_train",
                "20260928T094017Z_airline_v5_train"):
        meta = json.loads((ROOT / "runs" / run / "run.json").read_text())
        rows = [json.loads(x) for x in (ROOT / "runs" / run / "results.jsonl").read_text().splitlines() if x.strip()]
        rows = [r for r in rows if str(r["task_id"]) in shared]
        print(f"  {run[:16]} {meta['agent']} {meta['model']}: {sum(r['correct'] for r in rows)} of {len(rows)} shared train tasks")


def main() -> None:
    writes, convs = label()
    if "--models" in sys.argv:
        models()
        return
    if "--frequency" in sys.argv:
        frequency(writes, convs)
        return
    if "--folds" in sys.argv:
        folds(writes, convs)
        return
    if "--plans" in sys.argv:
        plans(writes, convs)
        return
    if "--bad" in sys.argv:
        for w in writes:
            if w["label"] != "bad":
                continue
            gold = [g for g in TASKS[w["task"]]["evaluation_criteria"]["actions"] if g["name"] == w["name"]]
            diff = sorted(k for k in set(w["arguments"]) | set(gold[0]["arguments"])
                          if w["arguments"].get(k) != gold[0]["arguments"].get(k)) if gold else ["no gold call of this tool"]
            print(w["split"], w["run"][:16], w["version"], w["task"], "*" if w["suspect"] else "", w["name"], ", ".join(diff))
        return
    for split in ("train", "test"):
        cs = [c for c in convs if c["split"] == split]
        ws = [w for w in writes if w["split"] == split]
        print(f"== {split}: {len(cs)} conversations from {len({c['run'] for c in cs})} runs, {sum(c['passed'] for c in cs)} passed")
        print("  modes", dict(collections.Counter(c["mode"] for c in cs)))
        print("  writes", dict(collections.Counter(w["label"] for w in ws)),
              "bad in passed conversations", sum(w["label"] == "bad" and w["passed"] for w in ws),
              "bad on suspect tasks", sum(w["label"] == "bad" and w["suspect"] for w in ws))
        print("  transfers", sum(c["transfer"] for c in cs), "in passes", sum(c["transfer"] and c["passed"] for c in cs),
              "in missed-write failures", sum(c["mode"] == "missed_write_transfer" for c in cs))
        print(f"  per conversation: writes {len(ws) / len(cs):.2f}, transfers {sum(c['transfer'] for c in cs) / len(cs):.2f}")
    v3 = [c for c in convs if c["version"] == "v3" and c["split"] == "train"]
    print("== v3 train:", len(v3), dict(collections.Counter(c["mode"] for c in v3)))
    shape(writes, convs)


def shape(writes: list[dict], convs: list[dict]) -> None:
    """How train turns are shaped, and what a judge call would read (characters, not tokens)."""
    turns = multi_turns = replies = multi_replies = 0
    for c in (c for c in convs if c["split"] == "train"):
        msgs = json.loads((ROOT / "runs" / c["run"] / "traces" / f"{c['task']}.json").read_text())["messages"]
        n = 0
        for m in msgs + [{"role": "user"}]:
            if m.get("role") == "user":
                turns, multi_turns, n = turns + 1, multi_turns + (n >= 2), 0
            elif m.get("role") == "assistant" and m.get("tool_calls"):
                k = len(m["tool_calls"])
                n, replies, multi_replies = n + k, replies + 1, multi_replies + (k > 1)
    prefix = []
    for w in (w for w in writes if w["split"] == "train" and w["label"] != "errored"):
        msgs = json.loads((ROOT / "runs" / w["run"] / "traces" / f"{w['task']}.json").read_text())["messages"]
        prefix.append(sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or [])) for m in msgs[: w["msg"] + 1]))
    agent_in = [
        json.loads(line)["agent_input_tokens"]
        for run in {c["run"] for c in convs if c["split"] == "train"}
        for line in (ROOT / "runs" / run / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    print(f"== train shape: {multi_turns} of {turns} user turns hold 2+ tool calls; "
          f"{multi_replies} of {replies} replies with tool calls hold more than one")
    print(f"  judge input: policy {len(EXTRACT['policy']):,} chars, tool schemas {len(json.dumps(EXTRACT['tools'])):,} chars, "
          f"transcript at the median executed write {statistics.median(prefix):,.0f} chars")
    print(f"  agent input tokens per train conversation: median {statistics.median(agent_in):,.0f} over {len(agent_in)}")
    traces = [
        sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or [])) for m in
            json.loads((ROOT / "runs" / c["run"] / "traces" / f"{c['task']}.json").read_text())["messages"])
        for c in convs if c["split"] == "train"
    ]
    gold = [len(json.dumps(TASKS[t]["evaluation_criteria"])) for t in SPLIT["train"]]
    print(f"  golden-answer input: whole trace median {statistics.median(traces):,.0f} chars (p90 {sorted(traces)[int(0.9 * len(traces))]:,}), "
          f"a task's gold criteria median {statistics.median(gold):,.0f} chars")


if __name__ == "__main__":
    main()
