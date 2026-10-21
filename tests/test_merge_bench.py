"""s30 (the person's call, 20 Oct 2026): the merge bench and options A, B and C.

r18's merges kept wrong steps and r19's removed right ones, because both settle a step by who
wrote it, not by the documents. The bench replays one job's merges with no research under each
option. No model is called: replies are scripted, or r19's recorded ones are fed back.
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path
from typing import Any

import pytest

from tau2_loop.config import ROOT
from tau2_loop.data import documents as kb_docs
from tau2_loop.workflows import bench, library, merge_opts, rag_agent, rubric, store
from test_rag_agent import CASH_BACK_DOC, OLD, WHEN, D, Reply

J = "filing_a_debit_card_transaction_dispute"
R19_088 = "20261015T115836Z_banking_knowledge_r19_task_088"
R18_088 = "20261013T052114Z_banking_knowledge_r18_task_088"
needs_docs = pytest.mark.skipif(not kb_docs.available(D), reason="tau2's documents are not here")


def _events(sid: str) -> list[dict[str, Any]]:
    p = rag_agent.RAG_RUNS_DIR / sid / "events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


@needs_docs
def test_r19s_recorded_replies_still_rebuild_r19s_versions_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: the merge code s30 changed is r19's when no option is set. r19 task_088's 15 recorded
    model and reviewer replies, fed back in order, commit the same five jobs r19 committed."""
    import dataclasses

    monkeypatch.setattr(store, "LIBRARY_DIR", tmp_path / "lib")
    ev = _events(R19_088)
    li = next(i for i, e in enumerate(ev) if e["kind"] == "locked")
    queue = [e for e in ev[li:] if e["kind"] in ("model", "review")]

    def canned(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        e = queue.pop(0)
        assert (tools is None and len(messages) == 1) == (e["kind"] == "review")
        res = types.SimpleNamespace(input_tokens=0, output_tokens=0, cache_read=0, duration_ms=0)
        text = e["reply"] if e["kind"] == "review" else e.get("text")
        calls = [
            {"id": c["id"], "name": c["name"], "arguments": c["args"]} for c in e.get("calls") or []
        ]
        return types.SimpleNamespace(content=text, tool_calls=calls, result=res)

    agent = dataclasses.replace(rag_agent.load(D, "r19"), name="r19x")
    d = store.LIBRARY_DIR / D / "r19x"
    d.mkdir(parents=True)
    (d / "seed.json").write_text((ROOT / "rag_library" / D / "r19" / "seed.json").read_text())
    system = (rag_agent.RAG_RUNS_DIR / R19_088 / "system.md").read_text()
    src = rag_agent.session(D, R18_088)
    replay = rag_agent._replay(src, system, rag_agent.SCRIPT_OPENING)
    run = tmp_path / "run"
    run.mkdir()
    meta = rag_agent.SessionMeta(
        id=f"20990101T000000Z_{D}_r19x_task_088",
        domain=D,
        agent="r19x",
        task_id="task_088",
        model="m",
        effort="high",
        retrieval="alltools_minilm",
        started="20990101T000000Z",
        source="remerge",
        drafts_of=R18_088,
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "KB_search_bm25", "description": "", "parameters": {}},
        }
    ]
    meta = rag_agent.run_session(
        run,
        meta,
        agent,
        system,
        tools,
        src["events"][0]["question"],
        canned,
        lambda n, a: "(not run)",
        bench.policy_of(system),
        rag_agent.SCRIPT_OPENING,
        replay,
    )
    assert meta.status == "done" and queue == []
    committed = json.loads((run / "output.json").read_text())["lock"]["committed"]
    r19 = json.loads((rag_agent.RAG_RUNS_DIR / R19_088 / "output.json").read_text())["lock"][
        "committed"
    ]
    assert set(committed) == set(r19) and len(committed) == 5
    for job, v in committed.items():
        mine = json.loads((d / "versions" / job / f"{v}.json").read_text())
        theirs = json.loads(
            (ROOT / "rag_library" / D / "r19" / "versions" / job / f"{r19[job]}.json").read_text()
        )
        assert mine == theirs, job


@needs_docs
def test_d19_and_d20_catch_r19s_head_that_lists_the_filing_tool_and_never_files() -> None:
    v3 = json.loads((ROOT / "rag_library" / D / "r19" / "versions" / J / "3.json").read_text())
    v0 = library._session_job("20261009T102616Z_banking_knowledge_r8_task_084", J)
    assert v0 is not None
    assert "file_debit_card_transaction_dispute_6281" in rubric.uncalled_writes(D, v3)
    assert rubric.uncalled_writes(D, v0) == []
    assert any(m["where"].startswith("tools: file_debit_card") for m in rubric.dangling(v3))
    c = rubric.check(D, v3, [v0])
    assert any("no step calls it" in w for w in rubric.gate(c, ("D20",), draft_base=True))
    assert not any("no step calls it" in w for w in rubric.gate(c, (), draft_base=True))


QUOTED = {
    **OLD["steps"][1],
    "quote": WHEN,
    "doc": CASH_BACK_DOC,
}  # a step that stands on its document
OLD_Q = {**OLD, "steps": [OLD["steps"][0], QUOTED]}


@needs_docs
def test_a_carries_a_sourced_step_the_merge_drops_without_an_account() -> None:
    protected = merge_opts.sourced_steps(D, OLD_Q, "")
    assert list(protected) == ["give"]  # verify has no quote: it needs no account
    job = {**OLD_Q, "steps": [OLD_Q["steps"][0]]}
    row = {"job": job, "previous": [{"job": "cash_back_dispute", "workflow": OLD_Q}], "opt": {}}
    lines = merge_opts.option_lines(D, row, {"cash_back_dispute": protected}, {}, "")
    assert len(lines) == 1 and "step give" in lines[0]
    covered = {**row, "opt": {"accounted": [{"step": "give", "covered_by": "verify"}]}}
    assert merge_opts.option_lines(D, covered, {"cash_back_dispute": protected}, {}, "") == []
    elsewhere = {**row, "opt": {"accounted": [{"step": "give", "covered_by": "no_such_step"}]}}
    assert merge_opts.option_lines(D, elsewhere, {"cash_back_dispute": protected}, {}, "")
    made_up = {
        **row,
        "opt": {
            "accounted": [
                {
                    "step": "give",
                    "replaced_by": {"quote": "Not a sentence of it.", "doc": CASH_BACK_DOC},
                }
            ]
        },
    }
    assert merge_opts.option_lines(D, made_up, {"cash_back_dispute": protected}, {}, "")
    # still unaccounted after the send-back: carried over where it stood
    settle = {**row, "checks": {}, "why": []}
    rag_agent._settle(D, [settle], "", (), True, {"cash_back_dispute": protected})
    assert [s["id"] for s in job["steps"]] == ["verify", "give"]
    assert any("carried over" in x["why"] for x in settle["settled"])


@needs_docs
def test_a_does_not_carry_back_a_step_the_reviewer_removed() -> None:
    job = {**OLD_Q, "steps": list(OLD_Q["steps"])}
    review = {
        "verdicts": [{"step": "give", "clause": "a discrepancy", "verdict": "no", "why": "x"}]
    }
    row = {
        "job": job,
        "previous": [{"job": "cash_back_dispute", "workflow": OLD_Q}],
        "checks": {},
        "why": [],
        "review": review,
        "opt": {},
    }
    rag_agent._settle(
        D, [row], "", (), True, {"cash_back_dispute": merge_opts.sourced_steps(D, OLD_Q, "")}
    )
    assert [s["id"] for s in job["steps"]] == ["verify"]  # removed with a reason, not lost


def test_b_audit_removes_invalid_rewrites_and_protects_what_is_valid() -> None:
    old = {
        "job": "j",
        "steps": [
            {"id": "a", "do": "verify"},
            {"id": "b", "do": "use a flat 500"},
            {"id": "c", "do": "drop the dispute when they correct a figure"},
        ],
        "rules": [{"rule": "r one"}, {"rule": "r two"}],
    }
    verdicts = {
        "steps": {
            "a": {"verdict": "valid"},
            "b": {
                "verdict": "rewrite",
                "rewrite": {"id": "zzz", "do": "the lesser of the tier and the amount"},
            },
            "c": {"verdict": "invalid", "why": "no passage gives that choice"},
        },
        "rules": {"R2": {"verdict": "invalid", "why": "unsupported"}},
    }
    a = merge_opts.audited(old, verdicts)
    assert [(s["id"], s["audit"]) for s in a["shown"]["steps"]] == [
        ("a", "valid"),
        ("b", "rewritten"),
    ]
    assert a["protected"]["b"]["do"].startswith("the lesser") and a["protected"]["b"]["id"] == "b"
    assert [x.get("step") or x.get("rule") for x in a["removed"]] == ["c", "r two"]
    assert [r["rule"] for r in a["shown"]["rules"]] == ["r one"]
    assert "drop the dispute" not in json.dumps(a["shown"]) and "c:" in merge_opts.removed_text(
        {"j": a}
    ).replace("step c", "c:")
    # the reviewer's no does not remove a valid step kept as the audit left it, but does a changed one
    job = {"job": "j", "steps": [a["protected"]["a"], {**a["protected"]["b"], "do": "changed"}]}
    review = {"verdicts": [{"step": "a", "verdict": "no"}, {"step": "b", "verdict": "no"}]}
    row = {
        "job": job,
        "previous": [{"job": "j", "workflow": old}],
        "checks": {},
        "why": [],
        "review": review,
        "opt": {"accounted": [{"step": "c", "covered_by": "a"}]},
    }
    rag_agent._settle(D, [row], "", (), True, {"j": a["protected"]}, audited=True)
    assert [s["id"] for s in job["steps"]] == ["a"]


def test_c_flags_a_slot_with_two_values_and_keeps_what_the_verifier_settled() -> None:
    claims = [
        {
            "id": "c1",
            "origin": ["L:tier"],
            "case": "every dispute",
            "field": "liability tier",
            "value": "count from the statement",
        },
        {
            "id": "c2",
            "origin": ["D:tier"],
            "case": "Every dispute",
            "field": "liability tier",
            "value": "count from discovery",
        },
        {
            "id": "c3",
            "origin": ["D:ask"],
            "case": "every dispute",
            "field": "ask: card in hand",
            "value": "ask",
        },
    ]
    groups = merge_opts.slots(claims)
    assert merge_opts.flagged(groups) == ["every dispute · liability tier"]
    verdicts = {"c1": {"kind": "stated"}, "c2": {"kind": "stated"}, "c3": {"kind": "question"}}
    settled = [{"slot": "every dispute · liability tier", "keep": ["c1"]}]
    surv, rejected = merge_opts.survivors(claims, verdicts, settled, merge_opts.flagged(groups))
    assert [c["id"] for c in surv] == ["c1", "c3"] and "settled for another value" in rejected["c2"]
    written = {"steps": [{"id": "s1", "claims": ["c1", "c2"]}]}
    _, opt = merge_opts.take_fields(written)
    lines = merge_opts.coverage_lines(merge_opts.claims_from_opt(opt), surv, rejected)
    assert any(x.startswith("claim c3 survived") for x in lines)
    assert any(x.startswith("claim c2 was rejected") for x in lines)
    assert "claims" not in merge_opts.take_fields(written)[0]["steps"][0]


KEY = ROOT / "merge_bench" / D / "eval" / "keys" / f"{J}.json"


def _fake_ask(seen: list[str], name: str = J):  # type: ignore[no-untyped-def]
    """Plausible replies for every call a bench merge makes, recording each prompt."""

    def ask(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        text = str(messages[-1]["content"] or "")
        seen.append("\n".join(str(m.get("content") or "") for m in messages))
        first = str(messages[0]["content"] or "")
        if first.startswith("You review one workflow job"):
            return Reply(json.dumps({"verdicts": []}))
        if first.startswith("You audit one workflow job"):
            ids = re.findall(r'^   "id": "([^"]+)"', text, re.M)  # a step of the audited job
            return Reply(
                json.dumps({"steps": [{"step": i, "verdict": "valid"} for i in ids], "rules": []})
            )
        if first.startswith("You break workflow jobs"):
            origins = sorted(set(re.findall(r'"id": "((?:L|D):[^"]+)"', first)))
            return Reply(
                json.dumps(
                    {
                        "cases": ["every dispute"],
                        "claims": [
                            {
                                "id": f"c{k}",
                                "from": "library",
                                "origin": [o],
                                "case": "every dispute",
                                "field": o,
                                "value": o,
                            }
                            for k, o in enumerate(origins)
                        ],
                    }
                )
            )
        if first.startswith("You verify claims"):
            ids = re.findall(r"^- (c\d+) ", first, re.M)
            return Reply(
                json.dumps({"claims": [{"id": i, "kind": "stated"} for i in ids], "slots": []})
            )
        draft = (
            next(
                json.loads(m["content"])
                for m in reversed(messages)
                if m["role"] == "assistant"
                and m.get("content")
                and str(m["content"]).startswith("{")
            )
            if not first.startswith("You write one workflow job")
            else None
        )
        if first.startswith("You write one workflow job"):
            ids = re.findall(r"^- (c\d+) ", first, re.M)
            job = {
                "job": J,
                "steps": [{"id": "s1", "do": "verify the customer", "claims": ids}],
                "tools": [],
            }
            return Reply(json.dumps({"jobs": [job]}))
        out = {**draft["jobs"][0], "job": name, "into": [name], "changelog": []}
        return Reply(json.dumps({"jobs": [out], "documents": [], "open_questions": []}))

    return ask


@needs_docs
@pytest.mark.parametrize("opt", ["A", "B", "C", "D", "D-lean", "D2"])
def test_each_option_replays_the_four_merges_and_no_prompt_holds_the_answer_key(
    opt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "LIBRARY_DIR", store.LIBRARY_DIR)  # bench.run repoints it
    seen: list[str] = []
    out = bench.run(
        D, J, opt, 1, ask=_fake_ask(seen), tools=[], root=tmp_path, stamp="20990101T000000Z"
    )
    rec = json.loads((out / "run.json").read_text())
    assert rec["status"] == "done", rec.get("error")
    assert [m["task"] for m in rec["merges"]] == ["task_084", "task_088", "task_090", "task_086"]
    hs = bench.heads(out, J)
    assert hs[0]["version"] == 0 and len(hs) >= 2
    key = json.loads(KEY.read_text())
    texts = [f["fact"] for f in key["facts"]] + [t["rule"] for t in key["traps"]]
    joined = "\n".join(seen)
    assert seen and not any(t in joined for t in texts)
    # E5 plays it: r19's library with this run's last version of the job in place of r19's
    monkeypatch.setattr(bench, "RUNS_DIR", tmp_path)
    lib = library.entries(D, f"r19@{out.name}")
    r19 = library.entries(D, "r19")
    assert len(lib) == len(r19)
    mine = library.by_name(lib, J)
    assert mine is not None and mine["workflow"] == hs[-1]["workflow"]
    assert mine["session"] == f"{out.name} v{hs[-1]['version']}"
    assert {e["job"] for e in lib} == {e["job"] for e in r19}


def test_the_grader_is_checked_against_the_hand_labels_and_its_reply_must_cover_the_key() -> None:
    from tau2_loop.workflows import merge_eval

    k = json.loads(KEY.read_text())
    facts = [{"id": f["id"], "verdict": "missing"} for f in k["facts"]]
    traps = [{"id": t["id"], "present": t["id"] == "T1"} for t in k["traps"]]
    g = merge_eval.parse_grade(json.dumps({"facts": facts, "traps": traps, "conflicts": []}), k)
    assert g is not None and g["score"] == 0 and g["traps_present"] == ["T1"]
    assert merge_eval.parse_grade(json.dumps({"facts": facts[:3], "traps": traps}), k) is None
    cal = merge_eval.calibration(k, {"r19_v3": g})
    wrong = {r["cell"] for r in cal["rows"] if not r["agree"]}
    # the hand labels say r19 v3 decides PIN or signature and states T3 and T4; this grade says not
    assert wrong == {"K3", "T3", "T4"}
    assert cal["of"] == len(k["calibration"]["r19_v3"])


def test_d10_reads_a_job_that_changes_a_pin_and_closes_a_card() -> None:
    """s30: option A's 090 merge called reset_debit_card_pin_6284, a write that leaves no status,
    beside close_debit_card_4721, and D10 crashed sorting None among the statuses."""
    from tau2_loop.workflows import toolspec

    def call(sid: str, tool: str) -> dict[str, Any]:
        return {
            "id": sid,
            "call": "call_discoverable_agent_tool",
            "args": {"agent_tool_name": tool},
        }

    job = {
        "steps": [call("pin", "reset_debit_card_pin_6284"), call("close", "close_debit_card_4721")]
    }
    assert toolspec.unmet_status(D, job) == []


@needs_docs
def test_d_post_audit_rewrites_and_never_removes_a_step_the_gate_needs() -> None:
    """D: the post-merge auditor's "invalid" removes a step only if the job still passes the gate:
    the reviewer's binding removal took the filing step out of r19 086 and of A's and C's merges."""
    from tau2_loop.workflows import toolspec

    filing = "file_debit_card_transaction_dispute_6281"
    params = toolspec.tools(D)[filing]["required"]
    f = {
        "id": "f",
        "call": "call_discoverable_agent_tool",
        "args": {"agent_tool_name": filing, **{p: "x" for p in params}},
    }
    job = {
        "job": "j",
        "tools": [{"tool": filing, "who": "agent", "step": "f"}],
        "steps": [f, {"id": "x", "do": "say a rule no passage gives"}, {"id": "y", "do": "old"}],
    }
    reply = {
        "steps": [
            {"step": "f", "verdict": "invalid", "why": "one clause unsupported"},
            {"step": "x", "verdict": "invalid", "why": "no passage"},
            {
                "step": "y",
                "verdict": "rewrite",
                "rewrite": {"do": "new"},
                "why": "the passage says new",
            },
        ],
        "rules": [],
    }
    row = {"job": job, "previous": [{"job": "j", "workflow": dict(job)}], "checks": {}, "why": []}
    merge_opts.post_audit(
        D,
        "{job}{version}{documents}{tools}",
        [row],
        "",
        ("D20",),
        lambda m, s: json.dumps(reply),
        lambda e: None,
    )
    assert [s["id"] for s in job["steps"]] == ["f", "y"]
    assert job["steps"][1]["do"] == "new" and row["post_audit"]["kept_for_the_gate"] == ["f"]
    assert any("step x" in x["what"] for x in job["changelog"])


# ── s31: D-lean, the person's calls O1-O5 and O8 on .lavish/s31_merge-bench-token-use.html ──


def _lean_ask(seen: list[str]):  # type: ignore[no-untyped-def]
    """D-lean's replies: the merge turn removes an audited step with no account, adds one and
    writes a job the merge does not hold; the send-back changes the added step only, as edits."""
    base = _fake_ask([])
    n = [0]

    def ask(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        text = str(messages[-1]["content"] or "")
        if len(messages) == 1:  # the auditor's and the reviewer's own contexts
            return base(messages, tools, model, effort, tool_mode)
        seen.append(text)
        if text.startswith("The harness applied your edits"):
            key = f"lean_new_{n[0]}"
            edit = {"op": "change", "list": "steps", "key": key, "why": "the send-back"}
            edit["fields"] = {"do": "ask which card, then file"}
            return Reply(json.dumps({"jobs": [{"job": J, "into": [J], "edits": [edit]}]}))
        current = text.split("as the audit left them", 1)[1]
        n[0] += 1
        item = {"id": f"lean_new_{n[0]}", "do": "ask the customer which card the charge was on"}
        first, second = re.findall(r'^   "id": "([^"]+)"', current, re.M)[:2]
        edits = [
            {"op": "remove", "list": "steps", "key": first, "why": "no account given"},
            {"op": "add", "list": "steps", "after": second, "item": item, "why": "the draft"},
        ]
        other = {"job": "a_job_it_does_not_hold", "into": [], "steps": [], "changelog": []}
        return Reply(json.dumps({"jobs": [{"job": J, "into": [J], "edits": edits}, other]}))

    return ask


@needs_docs
def test_d_lean_writes_edits_sends_back_what_failed_and_reviews_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "LIBRARY_DIR", store.LIBRARY_DIR)
    seen: list[str] = []
    out = bench.run(D, J, "D-lean", 1, ask=_lean_ask(seen), tools=[], root=tmp_path, stamp="1")
    rec = json.loads((out / "run.json").read_text())
    assert rec["status"] == "done", rec.get("error")
    assert (rec["merge_write"], rec["merge_steps"], rec["send_back"]) == ("edits", 0, "failed")
    assert all(m["taken"] for m in rec["merges"])
    for m in rec["merges"]:
        ev = _session_events(out, m["session"])
        users = [e["text"] for e in ev if e["kind"] == "user"]
        # O4: the merge prompt is the question itself, with no turns to search first
        assert not any(rag_agent.MERGE_WRITE_NOW in u for u in users)
        assert users[0].startswith("The library is holding") and len(users) == 2
        # O3: the job it does not hold is dropped before the checks and the reviewer
        notes = [e["text"] for e in ev if e["kind"] == "note"]
        assert any(n.startswith("Dropped 1 written job") for n in notes)
        reviews = [e for e in ev if e["kind"] == "review"]
        assert not any("a_job_it_does_not_hold" in e["text"] for e in reviews)
        # O5: the reviewer reads the first reply only; the auditor reads the last
        back = next(
            i
            for i, e in enumerate(ev)
            if e["kind"] == "user"
            and e is not ev[0]
            and str(e["text"]).startswith("The harness applied")
        )
        assert reviews and all(ev.index(e) < back for e in reviews)
        assert any(e.get("stage") == "post_audit" for e in ev if e["kind"] == "stage")
        out_j = json.loads((out / "sessions" / m["session"] / "output.json").read_text())
        merge = next(x for x in out_j["merge"] if x["job"] == J)
        assert merge["written"]["edits"] == 1  # the second reply's one edit
    # O2: the second reply's edit applied to the job as the first left it, and both changelogs
    # stand; the step removed with no account was carried back (O1's accounts read by code)
    head = bench.heads(out, J)[-1]["workflow"]
    new = {s["id"]: s["do"] for s in head["steps"] if str(s.get("id")).startswith("lean_new")}
    assert new == {f"lean_new_{k}": "ask which card, then file" for k in (1, 2, 3, 4)}
    last = json.loads((out / "sessions" / rec["merges"][-1]["session"] / "output.json").read_text())
    record = next(x for x in last["merge"] if x["job"] == J)
    log = [str(x.get("what")) + " " + str(x.get("why")) for x in record["changelog"]]
    assert any("steps lean_new_4" in x and "the draft" in x for x in log)
    assert any("the send-back" in x for x in log)
    assert any("carried over as it was" in x for x in log)
    assert not any('"audit":' in json.dumps(s) for s in head["steps"])
    assert sum(t.startswith("The harness applied your edits") for t in seen) == 4


def _session_events(out: Path, sid: str) -> list[dict[str, Any]]:
    p = out / "sessions" / sid / "events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_a_remove_edit_accounts_for_the_step_and_a_second_reply_keeps_the_first() -> None:
    entry = {
        "edits": [
            {"op": "remove", "list": "steps", "key": "a", "covered_by": "b"},
            {"op": "remove", "list": "steps", "key": "c", "quote": "q", "doc": "doc_1"},
            {"op": "remove", "list": "steps", "key": "d", "why": "nothing takes its place"},
            {"op": "remove", "list": "rules", "key": "a rule", "quote": "q"},
        ]
    }
    assert merge_opts.edit_accounts(entry) == [
        {"step": "a", "covered_by": "b"},
        {"step": "c", "replaced_by": {"quote": "q", "doc": "doc_1"}},
    ]
    joined = merge_opts.join_opts({"accounted": [1], "marks": {"s": 1}}, None, {"accounted": [2]})
    assert joined == {"accounted": [1, 2], "marks": {"s": 1}}
    ok = {"job": {"job": J}, "why": ["a line"], "blocked": False}
    failed = {"job": {"job": J}, "why": ["edits: edit 1: no item"], "blocked": True, "written": {}}
    notes: list[dict[str, Any]] = []
    rows = rag_agent._join_rounds([ok], [failed], notes.append)
    assert rows[0]["why"] == ["a line"] and rows[0]["second_reply"] == failed["why"]
    assert "did not apply" in notes[0]["text"]
    fixed = {"job": {"job": J}, "why": [], "blocked": False}
    assert rag_agent._join_rounds([ok], [fixed], notes.append) == [fixed]


@needs_docs
def test_upto_stops_a_run_for_scoring_and_it_resumes_where_it_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "LIBRARY_DIR", store.LIBRARY_DIR)
    ask = _lean_ask([])
    out = bench.run(D, J, "D-lean", 1, ask=ask, tools=[], root=tmp_path, stamp="1", upto=2)
    rec = json.loads((out / "run.json").read_text())
    assert rec["status"] == "stopped" and "O8" in rec["error"] and len(rec["merges"]) == 2
    assert "limit" not in rec["error"].lower()  # the resume command takes it
    bench.run(D, J, "D-lean", 1, ask=ask, tools=[], resume=out, why="scored after two")
    rec = json.loads((out / "run.json").read_text())
    assert rec["status"] == "done" and [m["task"] for m in rec["merges"]] == [
        "task_084",
        "task_088",
        "task_090",
        "task_086",
    ]
    assert [v["version"] for v in bench.heads(out, J)] == [0, 1, 2, 3, 4]


def test_a_second_grader_never_reads_the_first_graders_cached_replies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.workflows import merge_eval

    monkeypatch.setattr(merge_eval, "CACHE", tmp_path)
    opus = merge_eval.Grader(lambda text: "opus says", "")
    sonnet = merge_eval.Grader(lambda text: "sonnet says", "claude-sonnet-5-5-high")
    assert merge_eval._cached("grade", "same prompt", opus, None) == "opus says"
    assert merge_eval._cached("grade", "same prompt", sonnet, None) == "sonnet says"
    assert merge_eval._cached("grade", "same prompt", opus, None) == "opus says"
    assert {p.parent.name for p in tmp_path.rglob("*.json")} == {
        "grade",
        "grade@claude-sonnet-5-5-high",
    }
    assert merge_eval.eval_name(sonnet) == "eval@claude-sonnet-5-5-high.json"
    assert merge_eval.eval_name(lambda text: None) == "eval.json"


# ── s33: the second job, pin_lock_investigation_protocol (the person's call on s33) ──

PIN = "pin_lock_investigation_protocol"
PIN_KEY = ROOT / "merge_bench" / D / "eval" / "keys" / f"{PIN}.json"


@needs_docs
def test_d_replays_the_pin_lock_jobs_one_merge_and_no_prompt_holds_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.workflows import merge_eval

    monkeypatch.setattr(store, "LIBRARY_DIR", store.LIBRARY_DIR)
    seen: list[str] = []
    out = bench.run(D, PIN, "D", 1, ask=_fake_ask(seen, PIN), tools=[], root=tmp_path, stamp="1")
    assert out.name == f"1_{D}_{PIN}_D_t1"  # a job other than the first is named in its folder
    rec = json.loads((out / "run.json").read_text())
    assert rec["status"] == "done", rec.get("error")
    assert [m["task"] for m in rec["merges"]] == ["task_090"]
    hs = bench.heads(out, PIN)
    # the bench starts from r17's version, the baseline the grader is calibrated on
    assert hs[0]["workflow"] == merge_eval.baseline_version(
        D, PIN, merge_eval.baselines_of(D, PIN)["r17_v0"]
    )
    k = json.loads(PIN_KEY.read_text())
    texts = [f["fact"] for f in k["facts"]] + [t["rule"] for t in k["traps"]]
    assert seen and not any(x in "\n".join(seen) for x in texts)
    assert merge_eval.key_write(D, PIN) == "reset_debit_card_pin_6284"


def test_the_pin_lock_key_is_train_gold_only_and_its_labels_cover_every_cell() -> None:
    from tau2_loop.workflows import merge_eval

    k = json.loads(PIN_KEY.read_text())
    ids = {f["id"] for f in k["facts"]} | {t["id"] for t in k["traps"]}
    assert ids == {"K1", "K2", "K3", "K4", "K5", "K6", "K7", "T1", "T2", "T4"}  # T3 dropped (s33)
    split = json.loads((ROOT / "data" / "splits" / f"{D}.json").read_text())
    test_ids = set(split["test"])
    golds = " ".join(f["gold"] for f in k["facts"])
    assert not any(re.search(rf"\b{tid[-3:]}\b", golds) for tid in test_ids)
    assert set(merge_eval.baselines_of(D, PIN)) == {"r17_v0", "r18_v1", "r19_v1"}
    for name in ("r17_v0", "r18_v1", "r19_v1"):
        assert set(k["calibration"][name]) == ids
    assert k["calibration"]["r19_v1"]["K3"] == "wrong"
    facts = [
        {"id": i, "verdict": k["calibration"]["r19_v1"][i]} for i in sorted(ids) if i[0] == "K"
    ]
    traps = [
        {"id": i, "present": k["calibration"]["r19_v1"][i] == "present"}
        for i in sorted(ids)
        if i[0] == "T"
    ]
    g = merge_eval.parse_grade(json.dumps({"facts": facts, "traps": traps}), k)
    cal = merge_eval.calibration(k, {"r19_v1": g})
    assert g is not None and cal["agree"] == cal["of"] == len(ids)
