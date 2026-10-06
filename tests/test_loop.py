"""The loop's mechanics, offline: the prompt the optimiser gets, and one cycle with a fake optimiser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tau2_loop.loop.run as loop_run
from tau2_loop.agent.versions import load_version
from tau2_loop.eval import replay
from tau2_loop.eval.results import TaskResult, read_results
from tau2_loop.eval.runner import RunMeta, list_runs, load_run
from tau2_loop.loop.optimiser import OptimiserOutput, build_prompt, condense_trace, failure_details

MOCK_RUN = next((m.run_id for m in list_runs("mock") if m.summary), None)


@pytest.fixture(autouse=True)
def mlflow_writes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The registry's alias mirror, the prompt registry and the cycle log reach the central
    MLflow whenever it is up; no test writes there. Returns the cycles that would be logged."""
    from tau2_loop.tracking import mlflow_log
    from tau2_loop.tracking import registry as reg

    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(reg, "_mirror_alias", lambda *a, **k: None)
    monkeypatch.setattr(reg, "register_prompt", lambda *a, **k: None)
    monkeypatch.setattr(mlflow_log, "log_cycle", lambda entry: logged.append(dict(entry)))
    return logged


@pytest.mark.skipif(MOCK_RUN is None, reason="no committed mock run")
def test_optimiser_prompt_reads_the_committed_mock_run() -> None:
    meta, results = load_run(str(MOCK_RUN))
    failures = [r for r in results if r.correct is False]
    assert failures, "the committed mock run has failures to diagnose"
    champion = load_version("mock", "v0")
    prompt = build_prompt(champion, "v1", meta.run_id, failures)
    assert "agents/mock/v1/" in prompt and "diagnosis.json" in prompt
    for r in failures:
        assert f"## Task {r.task_id}" in prompt
    assert "### user" in prompt and "### agent" in prompt
    assert "McNemar" in prompt
    trace = Path("runs") / meta.run_id / "traces" / failures[0].trace
    details = failure_details(trace)
    assert "reward 0.0" in details
    assert "termination_reason" in condense_trace(trace)


V6_TRAIN = "20260928T101605Z_airline_v6_train"


@pytest.mark.skipif(not replay.available(), reason="tau2 is not installed")
def test_the_optimiser_reads_the_database_difference_and_the_actions_against_the_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each failure the optimiser reads carries what tau2 graded, rebuilt: every differing field
    and why, and each expected action matched or missing beside the agent's calls."""
    from tau2_loop.loop.optimiser import build_context, build_diagnose_prompt

    meta, results = load_run(V6_TRAIN)
    failures = [r for r in results if r.correct is False]
    champion = load_version("airline", "v6")
    ctx = build_context(champion, V6_TRAIN, failures)
    block = {b.split("\n")[0]: b for b in ctx.blocks}
    assert sorted(block) == [f"## Task {t} (trial 1)" for t in ("19", "22", "39", "44")]
    for b in block.values():
        assert "DATABASE DIFFERENCE" in b and "EXPECTED ACTIONS AGAINST THE AGENT'S CALLS" in b
        assert "ACTION is not in this task's reward basis" in b
        assert "expected action MISSING" not in b  # tau2's bare list, replaced
    # 44: the agent paid the fare difference from the gift card; gold charges the credit card
    b = block["## Task 44 (trial 1)"]
    assert "reservations.H8Q05L — wrong arguments" in b
    assert "users.sophia_silva_7557 — wrong write" in b
    assert 'payment_id: the agent "gift_card_5094406", expected "credit_card_4196779"' in b
    assert "MISSING 44_15 search_direct_flight (read, not graded)" in b
    # 39: the third cancellation was never made
    b = block["## Task 39 (trial 1)"]
    assert "reservations.MSJ4OA — missed write: gold's 39_10 cancel_reservation changes it" in b
    assert 'status: before not set · the agent left not set · gold expects "cancelled"' in b
    # both modes read the guide and must name the graded difference in each diagnosis
    for prompt in (
        build_prompt(champion, "v7", V6_TRAIN, failures, ctx),
        build_diagnose_prompt(champion, "v7", failures, ctx),
    ):
        assert "# Reading a failure" in prompt and '"graded_difference"' in prompt
        assert "the writes count, a missed lookup does not" in prompt

    # without tau2 the optimiser still reads tau2's own matched/missing list
    monkeypatch.setattr(replay, "available", lambda: False)
    b = build_context(champion, V6_TRAIN, failures).blocks[0]
    assert "DATABASE DIFFERENCE" not in b and "expected action MISSING" in b


V3_TRAIN = ("20260928T060029Z_airline_v3_train", "20260928T093954Z_airline_v3_train")


@pytest.mark.skipif(not replay.available(), reason="tau2 is not installed")
def test_the_optimiser_reads_every_run_of_the_champion_and_what_its_challengers_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record as it stood before cycle 6: v3's failures and, beside them, both runs of v3 and
    the challengers v4, v5 and v6 task by task. v4's "6 fixed" against v3's first run is 2 against
    both, since v3 itself flipped on the other four; what v4 and v6 broke comes graded."""
    import tau2_loop.eval.runner as runner
    import tau2_loop.loop.optimiser as opt
    from tau2_loop.loop.ledger import read_ledger
    from tau2_loop.loop.optimiser import build_context, build_diagnose_prompt
    from tau2_loop.loop.run import _read_failures

    # the runs and cycles there were then: every later one adds a column
    every_run, ledger = runner.list_runs, read_ledger("airline")[:5]
    monkeypatch.setattr(
        runner, "list_runs", lambda d=None: [m for m in every_run(d) if m.run_id < "20261002"]
    )
    monkeypatch.setattr(opt, "read_ledger", lambda d: ledger)
    champion = load_version("airline", "v3")
    meta, results = load_run(V3_TRAIN[1])
    failures = _read_failures("airline", results)
    ctx = build_context(champion, meta.run_id, failures)
    rec = ctx.record
    assert f"- v3·1 = runs/{V3_TRAIN[0]}" in rec and f"- v3·2 = runs/{V3_TRAIN[1]}" in rec
    assert "| task | v3·1 | v3·2 | v4 | v5 | v6 | reading |" in rec
    row = {
        ln.split(" | ")[0][2:]: ln for ln in rec.splitlines() if ln[:2] == "| " and ln[2].isdigit()
    }
    assert (
        row["23"] == "| 23 | ✗ | ✗ | ✓ | ✗ | ✓ | v3 fails it in every run: a real fix by v4, v6 |"
    )
    assert row["22"].endswith("v3 passes it in every run: a real break by v4, v6 |")
    assert row["19"].endswith(
        "v3 itself passes and fails it: a move here is as likely luck as a change |"
    )
    assert "The other 12 tasks passed in every run." in rec
    # every run here used tau2's own customer and the real date
    assert rec.count(" trial) †") == 5 and "† ran before the simulation rules of 2 Oct 2026" in rec

    v4 = rec.split("## v4 —")[1].split("## v5 —")[0]
    assert "Real fixes (every run of `v3` fails them, v4 passes): 23, 39." in v4
    assert "Real breaks (every run of `v3` passes them, v4 fails): 6, 20, 22, 47." in v4
    assert "Where `v3` itself flips (19 ✓, 24 ✓, 25 ✓, 33 ✓, 35 ✓ here)" in v4
    assert "CRITICAL — DO NOT OVERRIDE KNOWN PROFILE DATA" in v4  # its change log, edit by edit
    broke = v4.split("### v4 broke task 22")[1]
    assert "reservations.FQ8APE — missed write" in broke and "TRANSCRIPT:" in broke
    # a model swap has no surface to copy: its breaks come graded, without the conversation
    v5 = rec.split("## v5 —")[1].split("## v6 —")[0]
    assert "A model swap (model: sonnet → opus)" in v5 and "TRANSCRIPT:" not in v5
    assert "(task 6's scenario is above)" in v5
    # the conversations it names are the ones the fence lets the session read
    for rid, t in (("20260928T075613Z_airline_v4_train", "22"), (V3_TRAIN[0], "19")):
        assert (Path("runs") / rid / "traces" / f"{t}.json").resolve() in ctx.read_traces
    for prompt in (
        build_prompt(champion, "v7", meta.run_id, failures, ctx),
        build_diagnose_prompt(champion, "v7", failures, ctx),
    ):
        assert "# What the champion's challengers moved" in prompt
        assert '"carried_forward"' in prompt and '"dropped"' in prompt
        assert "are under # What the champion's challengers moved" in prompt  # the held block
    # a champion with one run and no challenger of its own has nothing to add
    v6 = load_version("airline", "v6")
    _, r6 = load_run(V6_TRAIN)
    assert build_context(v6, V6_TRAIN, [r for r in r6 if r.correct is False]).record == ""


def _rows(passes: set[str], ids: list[str]) -> list[TaskResult]:
    return [
        TaskResult(t, 1, 1.0 if t in passes else 0.0, t in passes, trace=f"{t}.json") for t in ids
    ]


def test_one_cycle_promotes_and_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake optimiser fixes five tasks and breaks none: promote, test runs compared, ledger complete."""
    import tau2_loop.config as cfg
    import tau2_loop.eval.runner as runner
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / d / "registry.json")
    ids = [str(i) for i in range(20)]
    calls: list[tuple[str, str, str]] = []

    def fake_eval(
        domain: str, agent: str, split: str = "train", **kw: Any
    ) -> tuple[RunMeta, list[TaskResult]]:
        calls.append((domain, agent, split))
        passes = set(ids[:10]) if agent == "v0" else set(ids[:15])
        rows = _rows(passes, ids)
        run_id = f"20260101T000000Z_{domain}_{agent}_{split}_{len(calls)}"
        d = tmp_path / "runs" / run_id
        d.mkdir(parents=True)
        from tau2_loop.eval.results import summarise, write_results

        write_results(d / "results.jsonl", rows)
        meta = RunMeta(
            run_id, domain, agent, "fp" + agent, "m", "u", "j", split, 20, 1, 3, 300, "t", "t"
        )
        meta.summary = summarise(rows).__dict__
        (d / "run.json").write_text(json.dumps(meta.__dict__))
        return meta, rows

    async def fake_optimiser(
        champion: Any, run_id: str, failures: list[TaskResult], model: str = "sonnet", **_: Any
    ) -> OptimiserOutput:
        assert len(failures) == 10
        return OptimiserOutput(
            "v1",
            {
                "diagnoses": [
                    {"task_id": f.task_id, "surface": "system.md", "change": "x"} for f in failures
                ],
                "prompt_diff_summary": "tightened confirmation rule",
                "helper_diff_summary": "",
                "expected_to_fix": [f.task_id for f in failures],
                "risks": [],
            },
            n_turns=5,
        )

    monkeypatch.setattr(loop_run, "run_eval", fake_eval)
    monkeypatch.setattr(loop_run, "run_optimiser", fake_optimiser)
    monkeypatch.setattr(
        loop_run,
        "load_version",
        lambda d, n: type("V", (), {"domain": d, "name": n, "fingerprint": "fp" + n})(),
    )
    monkeypatch.setattr(
        loop_run,
        "load_run",
        lambda rid: (
            RunMeta(rid, "airline", "v0", "fpv0", "m", "u", "j", "train", 20, 1, 3, 300, "t"),
            _rows(set(ids[:10]), ids),
        ),
    )

    import asyncio

    entry = asyncio.run(loop_run.run_cycle("airline", "v0", "sonnet", 3))
    o = entry["outcome"]
    assert o["verdict"] == "promote" and o["passes"] == "10 → 15"
    assert o["fixed"] == ids[10:15] and o["broken"] == []
    assert o["test_run"] and o["test_passes"] == "15/20"
    # champion, challenger, the challenger's test run, then the champion's (it had none)
    assert [(c[1], c[2]) for c in calls] == [
        ("v0", "train"),
        ("v1", "train"),
        ("v1", "test"),
        ("v0", "test"),
    ]
    tc = o["test_compare"]
    assert tc["passes"] == "10 → 15" and tc["fixed"] == ids[10:15] and tc["broken"] == []
    assert o["pass_1"] == "0.500 → 0.750"
    ledger = led.read_ledger("airline")
    assert len(ledger) == 1 and ledger[0]["outcome"]["verdict"] == "promote"
    assert ledger[0]["diagnoses"][0]["task_id"] == "10"
    # the cycle names the dataset profile it ran under, and that "sonnet" overrode its model
    from tau2_loop.loop.profiles import load_profile

    rec = ledger[0]["optimiser_profile"]
    assert rec["domain"] == "airline" and rec["fingerprint"] == load_profile("airline").fingerprint
    assert (rec["model"], rec["mode"], rec["tuned"]) == ("opus", "classic", False)
    assert rec["overrides"] == {"model": "sonnet"} and ledger[0]["optimiser_model"] == "sonnet"
    r = reg.read_registry("airline")
    assert r["champion"]["agent"] == "v1" and r["champion"]["passed"] == 15
    assert led.next_cycle_number("airline") == 2
    assert read_results(tmp_path / "runs" / o["challenger_run"] / "results.jsonl")[0].task_id == "0"


def test_a_domain_gated_on_test_reads_all_of_train_and_promotes_on_the_test_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Banking from 3 Oct 2026 (`GATE_ON_TEST`): the optimiser reads every train failure, the
    challenger's test run is compared with the champion's before the verdict, and that comparison
    decides. What the next optimiser reads is the moves on train; no test id reaches it. A run
    that would skip test is refused before a ledger line is written."""
    import asyncio

    import tau2_loop.config as cfg
    import tau2_loop.eval.runner as runner
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / d / "registry.json")
    train = [str(i) for i in range(20)]
    test = [f"t{i}" for i in range(20)]
    # v1 fixes 10–13 and breaks 0–1 on train: the old gate would hold it (two breaks, p ≈ 0.34);
    # on test it fixes six and breaks none
    passes = {
        ("v0", "train"): set(train[:10]),
        ("v1", "train"): set(train[2:14]),
        ("v0", "test"): set(test[:10]),
        ("v1", "test"): set(test[:16]),
    }
    calls: list[tuple[str, str]] = []

    def fake_eval(
        domain: str, agent: str, split: str = "train", **kw: Any
    ) -> tuple[RunMeta, list[TaskResult]]:
        calls.append((agent, split))
        ids = train if split == "train" else test
        rows = _rows(passes[(agent, split)], ids)
        run_id = f"20260101T000000Z_{domain}_{agent}_{split}_{len(calls)}"
        d = tmp_path / "runs" / run_id
        d.mkdir(parents=True)
        from tau2_loop.eval.results import summarise, write_results

        write_results(d / "results.jsonl", rows)
        meta = RunMeta(
            run_id, domain, agent, "fp" + agent, "m", "u", "j", split, 20, 1, 3, 300, "t", "t"
        )
        meta.task_ids = ids
        meta.summary = summarise(rows).__dict__
        (d / "run.json").write_text(json.dumps(meta.__dict__))
        return meta, rows

    seen: list[list[str]] = []

    async def fake_optimiser(
        champion: Any, run_id: str, failures: list[TaskResult], **_: Any
    ) -> OptimiserOutput:
        seen.append([f.task_id for f in failures])
        return OptimiserOutput("v1", {"diagnoses": [], "prompt_diff_summary": "x"}, n_turns=1)

    monkeypatch.setattr(loop_run, "GATE_ON_TEST", ("airline",))
    monkeypatch.setattr(loop_run, "halves", lambda d: None)
    monkeypatch.setattr(loop_run, "split_ids", lambda d, split: train if split == "train" else test)
    monkeypatch.setattr(loop_run, "run_eval", fake_eval)
    monkeypatch.setattr(loop_run, "run_optimiser", fake_optimiser)
    monkeypatch.setattr(
        loop_run,
        "load_version",
        lambda d, n: type("V", (), {"domain": d, "name": n, "fingerprint": "fp" + n})(),
    )

    with pytest.raises(ValueError, match="test run cannot be skipped"):
        asyncio.run(loop_run.run_cycle("airline", "v0", "sonnet", 3, run_test=False))
    assert calls == [] and led.read_ledger("airline") == []

    entry = asyncio.run(loop_run.run_cycle("airline", "v0", "sonnet", 3))
    o = entry["outcome"]
    # every train failure, not a half of them
    assert seen == [train[10:]]
    # the champion's train run, the challenger's, then both test runs, all before the verdict
    assert calls == [("v0", "train"), ("v1", "train"), ("v1", "test"), ("v0", "test")]
    assert o["gate_on"] == entry["gate_on"] == "test (20 tasks)"
    assert o["verdict"] == "promote" and o["rule"] == "mcnemar" and o["passes"] == "10 → 16"
    assert o["fixed"] == test[10:16] and o["broken"] == []
    assert o["test_compare"]["gated"] is True and o["test_compare"]["fixed"] == o["fixed"]
    assert o["train_passes"] == "10/20 → 12/20"
    assert o["read_fixed"] == train[10:14] and o["read_broken"] == ["0", "1"]
    champ = reg.read_registry("airline")["champion"]
    assert champ["agent"] == "v1" and champ["split"] == "train"  # its failures are the next read
    # the next optimiser reads the gate's counts and the train moves, never a test id
    history = led.render_history("airline", ["0"])
    assert "the gate's test passes 10 → 16 · on train 10/20 → 12/20" in history
    assert "fixed ['10', '11', '12', '13'] · broken ['0', '1']" in history
    assert "'t1" not in history and "reason" not in history
    assert led.prior_attempts("airline", "0")[0]["task_outcome"] == "broken"


def test_each_cycle_reads_its_dataset_profile_and_a_flag_overrides_it_for_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """s13 §5: the loop reads `optimisers/<domain>/` before every cycle, so a profile tuned after
    cycle 1 is cycle 2's; `--optimiser` / `--mode` replace its model and mode for one run,
    and each ledger entry names the profile's fingerprint and what was overridden. A mode
    neither knows is refused before the champion is evaluated."""
    import asyncio

    import tau2_loop.loop.profiles as profiles
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / d / "registry.json")
    monkeypatch.setattr(profiles, "OPTIMISERS_DIR", tmp_path / "optimisers")
    prof = tmp_path / "optimisers" / "retail"
    prof.mkdir(parents=True)
    (prof / "profile.yaml").write_text("model: sonnet\nmode: routing\n")
    ids = [str(i) for i in range(20)]
    seen: list[dict[str, Any]] = []
    champion_runs: list[str] = []

    async def optimiser(
        champion: Any, run_id: str, failures: list[TaskResult], **k: Any
    ) -> OptimiserOutput:
        seen.append(k)
        # the person tunes the profile before cycle 2 (in a real session an edit under
        # optimisers/ would reject this cycle: `optimiser.GUARDED`)
        (prof / "guide.md").write_text("Research first.\n")
        return OptimiserOutput("v1", {}, mode=k["mode"], error="no change", rejected=True)

    def champion_run(d: str, a: str, c: int, t: int = 1) -> tuple[RunMeta, list[TaskResult]]:
        champion_runs.append(a)
        return (
            RunMeta("r0", d, a, "fp", "m", "u", "j", "train", 20, 1, 3, 300, "t"),
            _rows(set(ids[:10]), ids),
        )

    monkeypatch.setattr(loop_run, "run_optimiser", optimiser)
    monkeypatch.setattr(loop_run, "_champion_run", champion_run)
    monkeypatch.setattr(
        loop_run,
        "load_version",
        lambda d, n: type("V", (), {"domain": d, "name": n, "fingerprint": "fp"})(),
    )

    asyncio.run(loop_run.run_loop("retail", cycles=2, agent="v0"))
    assert [(k["model"], k["mode"]) for k in seen] == [("sonnet", "routing")] * 2
    assert seen[0]["profile"].guide == "" and seen[1]["profile"].guide == "Research first.\n"
    a, b = led.read_ledger("retail")
    assert (a["optimiser_model"], a["optimiser_mode"]) == ("sonnet", "routing")
    pa, pb = a["optimiser_profile"], b["optimiser_profile"]
    assert pa["domain"] == "retail" and pa["tuned"] and pa["overrides"] == {}
    assert pa["fingerprint"] != pb["fingerprint"] == profiles.fingerprint("retail")
    assert (pa["guide_chars"], pb["guide_chars"]) == (0, len("Research first."))

    asyncio.run(
        loop_run.run_loop("retail", cycles=1, agent="v0", optimiser_model="opus", mode="classic")
    )
    assert (seen[-1]["model"], seen[-1]["mode"]) == ("opus", "classic")
    c = led.read_ledger("retail")[-1]
    assert c["optimiser_model"] == "opus" and c["optimiser_profile"]["model"] == "sonnet"
    assert c["optimiser_profile"]["overrides"] == {"model": "opus", "mode": "classic"}

    with pytest.raises(ValueError, match="mode must be one of"):
        asyncio.run(loop_run.run_cycle("retail", "v0", mode="fancy"))
    assert len(champion_runs) == 3 and len(led.read_ledger("retail")) == 3


def test_guard_writes_rejects_a_sibling_version_dir_with_colliding_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """agents/mock/v1 vs agents/mock/v10: a string-prefix check would wrongly let v10 writes through."""
    import asyncio

    import tau2_loop.agent.versions as versions
    import tau2_loop.loop.optimiser as opt

    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    monkeypatch.setattr(opt, "AGENTS_DIR", tmp_path)
    (tmp_path / "mock" / "v0").mkdir(parents=True)
    (tmp_path / "mock" / "v0" / "system.md").write_text("policy")
    (tmp_path / "mock" / "v0" / "agent.yaml").write_text("model: haiku\n")
    champion = versions.load_version("mock", "v0")

    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, options: Any) -> None:
            captured["options"] = options

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self) -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(opt, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(opt, "require_live", lambda: None)
    monkeypatch.setattr(opt, "subscription_env", lambda: {})

    asyncio.run(opt.run_optimiser(champion, "run0", []))

    # no sub-agent: one ran past the session's end and its diagnosis was never written (s15)
    assert captured["options"].tools == ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    guard_writes = captured["options"].hooks["PreToolUse"][0].hooks[0]
    new_dir = tmp_path / "mock" / "v1"
    sibling = tmp_path / "mock" / "v10" / "evil.py"
    allowed = new_dir / "system.md"

    denied = asyncio.run(guard_writes({"tool_input": {"file_path": str(sibling)}}, None, None))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    ok = asyncio.run(guard_writes({"tool_input": {"file_path": str(allowed)}}, None, None))
    assert ok == {}


def test_broken_trace_paths_map_each_task_to_its_own_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.eval.results import TaskResult, write_results
    from tau2_loop.loop.optimiser import _broken_trace_paths

    run_dir = tmp_path / "runs" / "20260101T000000Z_x"
    run_dir.mkdir(parents=True)
    write_results(
        run_dir / "results.jsonl",
        [
            TaskResult("3", 1, 0.0, False, trace="3.json"),
            TaskResult("7", 1, 0.0, False, trace="7.json"),
            TaskResult("9", 1, 1.0, True, trace="9.json"),
        ],
    )
    import tau2_loop.loop.optimiser as opt

    monkeypatch.setattr(opt, "RUNS_DIR", tmp_path / "runs")
    paths = _broken_trace_paths({"broken": ["3", "7"], "challenger_run": run_dir.name})
    assert paths == [
        f"runs/{run_dir.name}/traces/3.json",
        f"runs/{run_dir.name}/traces/7.json",
    ]


def test_rejected_optimiser_output_does_not_run_the_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / d / "registry.json")
    ids = [str(i) for i in range(20)]
    evals: list[str] = []

    async def bad_optimiser(*a: Any, **k: Any) -> OptimiserOutput:
        return OptimiserOutput(
            "v1", {}, error="optimiser changed files outside agents/retail/v1: ['src/x.py']"
        )

    monkeypatch.setattr(loop_run, "run_optimiser", bad_optimiser)
    monkeypatch.setattr(loop_run, "run_eval", lambda *a, **k: evals.append("eval"))
    monkeypatch.setattr(
        loop_run,
        "load_version",
        lambda d, n: type("V", (), {"domain": d, "name": n, "fingerprint": "fp"})(),
    )
    monkeypatch.setattr(
        loop_run,
        "_champion_run",
        lambda d, a, c, t=1: (
            RunMeta("r0", d, a, "fp", "m", "u", "j", "train", 20, 1, 3, 300, "t"),
            _rows(set(ids[:10]), ids),
        ),
    )
    import asyncio

    entry = asyncio.run(loop_run.run_cycle("retail", "v0", "sonnet", 3))
    assert entry["outcome"]["verdict"] == "rejected" and evals == []
    assert led.read_ledger("retail")[0]["outcome"]["verdict"] == "rejected"


def test_a_fork_inherits_its_sources_held_challengers_and_names_its_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v3 = v0's files on Sonnet: the optimiser sees v1 and v2 (held against v0) and says Sonnet."""
    import shutil

    import tau2_loop.agent.versions as versions
    import tau2_loop.loop.optimiser as opt
    from tau2_loop.config import AGENTS_DIR, LOOP_DIR
    from tau2_loop.loop import ledger as led

    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    for v in ("v0", "v1", "v2"):
        shutil.copytree(AGENTS_DIR / "airline" / v, tmp_path / "airline" / v)
    shutil.copyfile(LOOP_DIR / "airline" / "ledger.jsonl", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / "ledger.jsonl")
    v3 = versions.fork_version("airline", "v0", model="sonnet")
    assert v3.name == "v3"
    held = opt.held_challengers("airline", "v3")
    assert [h["challenger"] for h in held] == ["v1", "v2"]
    assert all(h["champion"] == "v0" for h in held)
    meta, results = load_run("20260915T132148Z_airline_v2_train")
    failures = [r for r in results if r.correct is False][:1]
    prompt = build_prompt(v3, "v4", meta.run_id, failures)
    assert (
        "The agent is Claude Sonnet 5" in prompt
        and "Claude Haiku 4.5) plays the customer" in prompt
    )
    assert "a fork of `v0`" in prompt and "agents/airline/v2/" in prompt
    assert "breaks none" in prompt


# ── make challenge: a hand-made version through the cycle's own gate ─────────

TRAIN = [f"tr{i}" for i in range(20)]
TEST = [f"te{i}" for i in range(20)]
# (agent, split) → the tasks that pass: v3 fails 6 train tasks, Opus fixes 3 and breaks none
PASSES = {
    ("v3", "train"): set(TRAIN[:14]),
    ("v5", "train"): set(TRAIN[:17]),
    ("v3", "test"): set(TEST[:15]),
    ("v5", "test"): set(TEST[:16]),
}


def _write_run(
    runs: Path,
    run_id: str,
    agent: Any,
    split: str,
    scored: bool = True,
    started_at: str = "",
    sim_rules: str | None = None,
) -> str:
    """A run folder as `run_eval` leaves it; `scored=False` is one still running (or dead), and
    `sim_rules="tau2"` one made before our rules (run.json has no `sim_rules`)."""
    from dataclasses import asdict

    from tau2_loop.eval.results import summarise, write_results
    from tau2_loop.eval.runner import SIM_RULES

    ids = TRAIN if split == "train" else TEST
    rows = _rows(PASSES[(agent.name, split)], ids)
    d = runs / run_id
    d.mkdir(parents=True)
    meta = RunMeta(
        run_id=run_id,
        domain="airline",
        agent=agent.name,
        fingerprint=agent.fingerprint,
        model="claude-sdk/" + agent.config.model,
        user_model="claude-sdk/claude-haiku-4-5",  # the loop reuses only runs with its customer (s16)
        judge_model="j",
        split=split,
        n_tasks=len(ids),
        trials=1,
        concurrency=3,
        seed=300,
        started_at=started_at or "2026-09-28T06:00:00+00:00",
        tau2_sha="t",
        task_ids=ids,
        split_version=2,
        sim_rules=None if sim_rules == "tau2" else SIM_RULES,
    )
    if scored:
        write_results(d / "results.jsonl", rows)
        meta.summary = asdict(summarise(rows))
        meta.finished_at = meta.started_at
    (d / "run.json").write_text(json.dumps(asdict(meta)))
    return run_id


def _challenge_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v5_train: str = "scored",
    champion_rules: str | None = None,
) -> dict[str, Any]:
    """agents/, runs/ and loop/ under tmp_path, shaped like airline today: v3 (Sonnet) is the
    champion with a train and a test run, v4 a loop challenger, v5 = `fork` of v3 on Opus.
    `v5_train` is v5's train run: "scored", "running" (started now, no summary), "dead"
    (no summary, two days old) or "none"; `champion_rules="tau2"` makes v3's runs before our
    simulation rules. run_eval is faked and records what it was asked."""
    from datetime import UTC, datetime, timedelta

    import tau2_loop.agent.versions as versions
    import tau2_loop.config as cfg
    import tau2_loop.eval.runner as runner
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    runs = tmp_path / "runs"
    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(cfg, "RUNS_DIR", runs)
    monkeypatch.setattr(runner, "RUNS_DIR", runs)
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / "loop" / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / "loop" / d / "registry.json")
    monkeypatch.setattr(loop_run, "split_ids", lambda d, s: {"train": TRAIN, "test": TEST}[s])

    for name, prompt in (
        ("v3", "Be an airline agent.\n{policy}\n"),
        ("v4", "Confirm first.\n{policy}\n"),
    ):
        vdir = tmp_path / "agents" / "airline" / name
        vdir.mkdir(parents=True)
        (vdir / "system.md").write_text(prompt)
        (vdir / "agent.yaml").write_text("model: sonnet\neffort: medium\n")
    v3 = versions.load_version("airline", "v3")
    v5 = versions.fork_version("airline", "v3", model="opus")
    assert v5.name == "v5"

    world: dict[str, Any] = {"calls": []}
    old = champion_rules
    world["v3_train"] = _write_run(
        runs, "20260928T060029Z_airline_v3_train", v3, "train", sim_rules=old
    )
    world["v3_test"] = _write_run(
        runs, "20260928T073602Z_airline_v3_test", v3, "test", sim_rules=old
    )
    reg.promote(world["v3_train"], kind="model swap")
    if v5_train != "none":
        started = datetime.now(UTC) - timedelta(days=2 if v5_train == "dead" else 0)
        world["v5_train"] = _write_run(
            runs,
            "20260928T090000Z_airline_v5_train",
            v5,
            "train",
            scored=v5_train == "scored",
            started_at=started.isoformat(),
        )

    def fake_eval(
        domain: str, agent: str, split: str = "train", **kw: Any
    ) -> tuple[RunMeta, list[TaskResult]]:
        world["calls"].append((agent, split))
        v = versions.load_version(domain, agent)
        run_id = f"20260928T1{len(world['calls']):05d}Z_{domain}_{agent}_{split}"
        _write_run(runs, run_id, v, split)
        return load_run(run_id)

    async def no_optimiser(*a: Any, **k: Any) -> OptimiserOutput:
        raise AssertionError("a challenge never runs the optimiser")

    monkeypatch.setattr(loop_run, "run_eval", fake_eval)
    monkeypatch.setattr(loop_run, "run_optimiser", no_optimiser)
    return world


def test_a_model_swap_challenge_reuses_its_runs_and_records_a_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mlflow_writes: list[dict[str, Any]]
) -> None:
    """v5 = v3's prompt on Opus: v5's train run exists, so only its test split runs; the gate,
    the ledger, the promotion and the test report are the loop's own, with no optimiser."""
    from tau2_loop.llm import model_label
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    world = _challenge_world(tmp_path, monkeypatch)
    entry = loop_run.run_challenge("airline", "v5")

    # (a) the champion's registry run, v5's train run and v3's test run are all reused
    assert world["calls"] == [("v5", "test")]
    o = entry["outcome"]
    assert entry["champion_run"] == world["v3_train"] and o["challenger_run"] == world["v5_train"]
    # (c) the gate's own verdict, in a ledger entry that names the swap and no optimiser
    assert o["verdict"] == "promote" and o["passes"] == "14 → 17"
    assert o["fixed"] == TRAIN[14:17] and o["broken"] == [] and o["still_failed"] == TRAIN[17:]
    [e] = led.read_ledger("airline")
    assert e["cycle"] == 1 and e["kind"] == "model swap" and e["forked_from"] == "v3"
    assert e["champion"] == "v3" and e["challenger"] == "v5"
    assert e["agent_yaml"] == ["model: sonnet → opus"]
    assert e["challenger_model"] == model_label("opus") and e["challenger_effort"] == "medium"
    assert e["optimiser_model"] is None and e["optimiser"] is None
    assert e["diagnoses"] == [] and e["expected_to_fix"] == []
    assert e["failed"] == TRAIN[14:] and e["trials"] == 1 and e["split_version"] == 2
    assert e["outcome"]["verdict"] == "promote" and e["outcome"]["reason"] == o["reason"]
    assert "eval_agent_in" in e["tokens"] and "optimiser_in" not in e["tokens"]
    # promoted as the gate promotes a loop challenger
    r = reg.read_registry("airline")
    assert r["champion"]["agent"] == "v5" and r["history"][-1]["kind"] == "gate"
    # (d) champion vs challenger on test, reported beside the verdict
    tc = e["outcome"]["test_compare"]
    assert tc["champion_run"] == world["v3_test"] and tc["passes"] == "15 → 16"
    assert tc["fixed"] == [TEST[15]] and tc["broken"] == []
    assert e["outcome"]["test_passes"] == "16/20"
    assert mlflow_writes[-1]["kind"] == "model swap"
    # the history the next optimiser session reads says what this cycle was
    history = led.render_history("airline", [])
    assert "v3 → v5, a model swap (model: sonnet → opus), no optimiser · verdict promote" in history


def test_the_gate_reruns_a_champion_run_made_before_our_simulation_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs under tau2's rules and ours never meet at the gate: the champion's train and test
    runs, both made before our rules, are played again, and the new train run is its record."""
    from tau2_loop.tracking import registry as reg

    world = _challenge_world(tmp_path, monkeypatch, champion_rules="tau2")
    entry = loop_run.run_challenge("airline", "v5")
    assert sorted(world["calls"]) == [("v3", "test"), ("v3", "train"), ("v5", "test")]
    assert entry["champion_run"] != world["v3_train"]
    assert entry["outcome"]["test_compare"]["champion_run"] != world["v3_test"]
    rebased = [
        h["run_id"]
        for h in reg.read_registry("airline")["history"]
        if h.get("event") == "promote" and h["kind"] == "re-baseline"
    ]
    assert rebased == [entry["champion_run"]]


def test_a_challenge_runs_train_only_when_no_run_of_those_bytes_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A train run still in flight is waited for, never duplicated; a dead one does not block."""
    from tau2_loop.loop import ledger as led

    world = _challenge_world(tmp_path, monkeypatch, v5_train="running")
    with pytest.raises(RuntimeError, match="no summary yet"):
        loop_run.run_challenge("airline", "v5")
    assert world["calls"] == [] and led.read_ledger("airline") == []

    world = _challenge_world(tmp_path / "dead", monkeypatch, v5_train="dead")
    entry = loop_run.run_challenge("airline", "v5", run_test=False)
    assert world["calls"] == [("v5", "train")]
    assert entry["outcome"]["verdict"] == "promote" and "test_run" not in entry["outcome"]


def test_a_challenge_refuses_the_champion_and_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from tau2_loop.loop import ledger as led

    world = _challenge_world(tmp_path, monkeypatch)
    agents = tmp_path / "agents" / "airline"
    shutil.copytree(agents / "v3", agents / "v6")  # the champion's bytes under another name
    with pytest.raises(ValueError, match="champion"):
        loop_run.run_challenge("airline", "v3")
    with pytest.raises(ValueError, match="bytes"):
        loop_run.run_challenge("airline", "v6")
    assert world["calls"] == [] and led.read_ledger("airline") == []


def test_an_ab_pair_reads_the_same_half_gates_on_the_other_and_crowns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two optimisers from one champion (s09 §6): each reads only the read half's failures and is
    gated on the gate half; both pass, the one with more gate passes takes the title."""
    import asyncio

    import tau2_loop.config as cfg
    import tau2_loop.eval.runner as runner
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / d / "registry.json")
    ids = [str(i) for i in range(20)]
    read, gate = ids[0::2], ids[1::2]
    passes = {"v0": set(ids[:10]), "v1": set(ids[:12]), "v2": set(ids[:14])}
    calls: list[tuple[str, str]] = []

    def fake_eval(
        domain: str, agent: str, split: str = "train", **kw: Any
    ) -> tuple[RunMeta, list[TaskResult]]:
        calls.append((agent, split))
        rows = _rows(passes[agent], ids)
        run_id = f"20260101T000000Z_{domain}_{agent}_{split}_{len(calls)}"
        d = tmp_path / "runs" / run_id
        d.mkdir(parents=True)
        from tau2_loop.eval.results import summarise, write_results

        write_results(d / "results.jsonl", rows)
        meta = RunMeta(
            run_id, domain, agent, "fp" + agent, "m", "u", "j", split, 20, 1, 3, 300, "t", "t"
        )
        meta.summary = summarise(rows).__dict__
        (d / "run.json").write_text(json.dumps(meta.__dict__))
        return meta, rows

    seen: list[dict[str, Any]] = []

    async def fake_optimiser(
        champion: Any, run_id: str, failures: list[TaskResult], **k: Any
    ) -> OptimiserOutput:
        seen.append({"failed": [f.task_id for f in failures], **k})
        name = "v1" if k["mode"] == "classic" else "v2"
        return OptimiserOutput(
            name, {"diagnoses": []}, mode=k["mode"], surfaces_changed=["system.md"]
        )

    monkeypatch.setattr(loop_run, "halves", lambda d: (read, gate))
    monkeypatch.setattr(
        loop_run, "GATE_ON_TEST", ()
    )  # s09's halved gate, before banking's moved to test
    monkeypatch.setattr(loop_run, "build_context", lambda *a: "ctx")
    monkeypatch.setattr(loop_run, "read_registry", lambda d: {"champion": {"agent": "v0"}})
    monkeypatch.setattr(loop_run, "run_eval", fake_eval)
    monkeypatch.setattr(loop_run, "run_optimiser", fake_optimiser)
    monkeypatch.setattr(
        loop_run,
        "load_version",
        lambda d, n: type("V", (), {"domain": d, "name": n, "fingerprint": "fp" + n})(),
    )
    monkeypatch.setattr(loop_run, "version_dir", lambda d, n: tmp_path / "agents" / d / n)
    monkeypatch.setattr(
        loop_run,
        "load_run",
        lambda rid: (
            RunMeta(
                rid, "banking_knowledge", "v0", "fpv0", "m", "u", "j", "train", 20, 1, 3, 300, "t"
            ),
            _rows(passes["v0"], ids),
        ),
    )

    entries = asyncio.run(loop_run.run_ab("banking_knowledge"))
    # both read the same five read-half failures, from one context; the second cannot see the first
    assert seen[0]["failed"] == seen[1]["failed"] == ["10", "12", "14", "16", "18"]
    assert [s["mode"] for s in seen] == ["classic", "routing"]
    assert seen[0]["hidden"] == [] and seen[1]["hidden"] == [
        tmp_path / "agents" / "banking_knowledge" / "v1"
    ]
    assert seen[0]["ctx"] == seen[1]["ctx"] == "ctx"
    a, b = (e["outcome"] for e in entries)
    # the gate reads only the gate half: the champion passes 5 of its 10
    assert a["gate_passes"] == {"champion": 5, "challenger": 6, "n": 10} and a["fixed"] == ["11"]
    assert b["gate_passes"]["challenger"] == 7 and b["verdict"] == "promote"
    assert a["verdict"] == "hold" and "A/B partner v2 took the title" in a["reason"]
    assert a["train_passes"] == "10/20 → 12/20"
    assert reg.read_registry("banking_knowledge")["champion"]["agent"] == "v2"
    ledger = led.read_ledger("banking_knowledge")
    assert [e["optimiser_mode"] for e in ledger] == ["classic", "routing"]
    # one profile for the pair, read once; its model by default, and the pair's second mode noted
    assert seen[0]["profile"] is seen[1]["profile"] and seen[0]["model"] == "opus"
    # banking's profile routes since s14, so the classic partner is the override
    assert [e["optimiser_profile"]["overrides"] for e in ledger] == [{"mode": "classic"}, {}]
    assert ledger[0]["experiment"]["pair"] == [1, 2] and ledger[0]["outcome"]["verdict"] == "hold"
    assert ledger[1]["gate_on"] == "gate half (10 of 20 train tasks)"
    # what the next optimiser reads: the read half's fixes, never a gate-half id
    assert b["read_fixed"] == ["10", "12"] and b["read_broken"] == []
    history = led.render_history("banking_knowledge", ["14"])
    assert "on the read half fixed ['10', '12']" in history
    assert "'11'" not in history and "'13'" not in history and "A/B partner" not in history


def test_a_routing_optimiser_writes_only_what_its_diagnosis_routed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnose, route, write, package (s09 §5): the writer gets only the routed files, the
    diagnosis is frozen between the two sessions, and a code surface that imports `os` is refused."""
    import asyncio

    import tau2_loop.agent.versions as versions
    import tau2_loop.loop.optimiser as opt

    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    monkeypatch.setattr(opt, "AGENTS_DIR", tmp_path)
    champ_dir = tmp_path / "airline" / "v0"
    champ_dir.mkdir(parents=True)
    (champ_dir / "system.md").write_text("You are an agent.\n{policy}")
    (champ_dir / "agent.yaml").write_text("model: haiku\n")
    champion = versions.load_version("airline", "v0")
    new_dir = tmp_path / "airline" / "v1"
    sessions: list[Any] = []

    class FakeClient:
        def __init__(self, options: Any) -> None:
            sessions.append(options)

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            if "**diagnosis** step" in prompt:
                assert (new_dir / ".context" / "policy.md").exists()
                (new_dir / "diagnosis.json").write_text(
                    json.dumps(
                        {
                            "diagnoses": [
                                {
                                    "task_id": "3",
                                    "class": "write-arguments",
                                    "surfaces": ["checks.py", "memory.py"],
                                }
                            ],
                            "surfaces": {
                                "checks.py": "origin is not destination",
                                "memory.py": "keep the user",
                            },
                        }
                    )
                )
            else:
                assert "`checks.py`, `memory.py`" in prompt
                (new_dir / "checks.py").write_text(
                    "def check_write(name, arguments, state):\n"
                    "    if arguments.get('origin') == arguments.get('destination'):\n"
                    "        return 'the destination cannot be the origin.'\n"
                    "    return None\n"
                )
                (new_dir / "memory.py").write_text(
                    "import os\n\ndef remember(state, name, arguments, result):\n    state['cwd'] = os.getcwd()\n"
                )
                (new_dir / "changes.json").write_text(
                    json.dumps({"changes": [{"file": "checks.py"}]})
                )

        async def receive_response(self) -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(opt, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(opt, "require_live", lambda: None)
    monkeypatch.setattr(opt, "subscription_env", lambda: {})

    out = asyncio.run(opt.run_optimiser(champion, "run0", [], mode="routing"))

    assert out.mode == "routing" and out.routed == ["checks.py", "memory.py"]
    assert out.surfaces_changed == ["checks.py", "memory.py"]
    assert out.rejected and "guard: memory.py imports os" in (out.error or "")
    d = json.loads((new_dir / "diagnosis.json").read_text())
    assert d["routed"] == ["checks.py", "memory.py"] and d["changes"] == [{"file": "checks.py"}]
    assert not (new_dir / ".context").exists() and not (new_dir / "changes.json").exists()
    # the diagnosis session may write diagnosis.json only; the writer only the routed files
    guard_diag, guard_write = (s.hooks["PreToolUse"][0].hooks[0] for s in sessions)
    deny = lambda g, p: asyncio.run(g({"tool_input": {"file_path": str(p)}}, None, None))  # noqa: E731
    assert (
        deny(guard_diag, new_dir / "system.md")["hookSpecificOutput"]["permissionDecision"]
        == "deny"
    )
    assert deny(guard_diag, new_dir / "diagnosis.json") == {}
    assert deny(guard_write, new_dir / "checks.py") == {}
    assert (
        deny(guard_write, new_dir / "system.md")["hookSpecificOutput"]["permissionDecision"]
        == "deny"
    )
    assert (
        deny(guard_write, new_dir / "diagnosis.json")["hookSpecificOutput"]["permissionDecision"]
        == "deny"
    )
    guard_reads = sessions[0].hooks["PreToolUse"][1].hooks[0]
    fenced = asyncio.run(
        guard_reads({"tool_input": {"file_path": "data/tasks/airline.json"}}, None, None)
    )
    assert fenced["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_checking_helper_by_importing_it_does_not_reject_the_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prompt asks the optimiser to verify helper.py by importing it, which leaves
    `__pycache__` in the version folder: airline cycle 6 (v7) was rejected for exactly that."""
    import asyncio
    import subprocess
    import sys

    import tau2_loop.agent.versions as versions
    import tau2_loop.loop.optimiser as opt

    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    monkeypatch.setattr(opt, "AGENTS_DIR", tmp_path)
    champ_dir = tmp_path / "airline" / "v0"
    champ_dir.mkdir(parents=True)
    (champ_dir / "system.md").write_text("You are an agent.\n{policy}")
    (champ_dir / "agent.yaml").write_text("model: haiku\n")
    champion = versions.load_version("airline", "v0")
    new_dir = tmp_path / "airline" / "v1"

    class FakeClient:
        def __init__(self, options: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            (new_dir / "helper.py").write_text("def on_reply(text):\n    return text\n")
            subprocess.run(
                [sys.executable, "-c", "import helper; assert helper.on_reply('x') == 'x'"],
                cwd=new_dir,
                check=True,
            )
            (new_dir / "diagnosis.json").write_text(json.dumps({"diagnoses": []}))

        async def receive_response(self) -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(opt, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(opt, "require_live", lambda: None)
    monkeypatch.setattr(opt, "subscription_env", lambda: {})

    out = asyncio.run(opt.run_optimiser(champion, "run0", [], mode="classic"))
    assert not out.rejected and out.error is None, out.error
    assert out.surfaces_changed == ["helper.py"] and not (new_dir / "__pycache__").exists()


BANKING_V1 = (
    "20261002T143917Z_banking_knowledge_v1_train",
    "20261002T160930Z_banking_knowledge_v1_test",
)
# v1 extended to split v3 (`make extend`): each joins the run above with the run of the new tasks
BANKING_V1_V3 = (
    "20261002T235638Z_banking_knowledge_v1_train",
    "20261003T003126Z_banking_knowledge_v1_test",
)
# v3 (s13: v1 with AllTools, native tool calls and the identity note), champion by the person's call
BANKING_V3 = (
    "20261004T023602Z_banking_knowledge_v3_train",
    "20261004T023605Z_banking_knowledge_v3_test",
)
# v4 (s14: `make optimise` from v3 at high effort with parallel calls), promoted by the gate on test
BANKING_V4 = (
    "20261004T222002Z_banking_knowledge_v4_train",
    "20261004T113341Z_banking_knowledge_v4_test",
)


def test_banking_v1_is_scored_on_split_v3_by_extension_and_nothing_else() -> None:
    """Banking's v0 is retired (3 Oct 2026); v1 is champion. Split v3 (60 / 37) added 12 train and
    12 test tasks, and v1's runs were extended rather than replayed: one run of each split, joined
    from v1's split v2 run and a run of only the new tasks. A gate or a challenge reuses exactly
    those, and none of v1's other runs reaches the optimiser or a comparison. Cycle 1's v2 was
    promoted by s09's halved gate, then re-decided on test when the gate moved there: held, and
    v1's title restored. Cycle 2's v3, a tool change, was held by the gate (6 → 6 on test) and
    then promoted by the person's call (4 Oct 2026): a harness without v1's defects. Cycle 3's v4,
    written by `make optimise` from v3 (Sonnet at high effort, every tool call of a reply kept),
    was promoted by the gate on test: 6 → 14, fixed 8, broke 0 (5 Oct 2026)."""
    from tau2_loop.agent.versions import base_version, lineage
    from tau2_loop.data.splits import read_split, split_ids
    from tau2_loop.loop.ledger import read_ledger
    from tau2_loop.loop.optimiser import champion_record, held_challengers
    from tau2_loop.tracking.registry import read_registry

    d = "banking_knowledge"
    reg = read_registry(d)
    assert reg["champion"]["run_id"] == BANKING_V4[0] and reg["challenger"] is None
    assert [(h["agent"], h["kind"]) for h in reg["history"] if h["event"] == "promote"] == [
        ("v1", "model swap"),
        ("v1", "re-baseline"),
        ("v2", "gate"),
        ("v1", "re-decided"),
        ("v3", "tool change"),
        ("v4", "gate"),
    ]
    e, e2, e3 = read_ledger(d)
    o3 = e3["outcome"]
    assert (e3["kind"], e3["champion"], e3["challenger"]) == ("optimised", "v3", "v4")
    assert o3["verdict"] == "promote" and o3["passes"] == "6 → 14" and o3["broken"] == []
    assert o3["test_run"] == BANKING_V4[1] and e3["optimised_from"] == "v3"
    assert len(e3["diagnoses"]) == 48 and e3["optimiser"]["mode"] == "routing"
    assert e3["agent_yaml"] == ["effort: medium → high", "parallel_calls: False → True"]
    o2 = e2["outcome"]
    assert (e2["kind"], e2["champion"], e2["challenger"]) == ("tool change", "v1", "v3")
    assert o2["verdict"] == "hold" and o2["passes"] == "6 → 6" and o2["test_run"] == BANKING_V3[1]
    o = e["outcome"]
    assert e["gate_on"] == o["gate_on"] == "test (37 tasks)" and o["verdict"] == "hold"
    assert o["passes"] == "6 → 5" and o["test_compare"]["gated"] is True
    assert o["test_compare"]["champion_run"] == BANKING_V1_V3[1]
    assert o["superseded"]["verdict"] == "promote" and o["superseded"]["passes"] == "2 → 7"
    assert o["train_passes"] == "2/60 → 17/60" and len(o["read_fixed"]) == 15
    assert base_version(d) == "v1" and base_version("airline") == "v0"
    # v1 says it was forked from v0; with v0's folder gone its lineage is its own
    assert lineage(d, "v1") == ["v1"] and lineage("airline", "v3") == ["v3", "v0"]
    s = read_split(d)
    for split, old, run in zip(("train", "test"), BANKING_V1, BANKING_V1_V3, strict=True):
        meta, rows = load_run(run)
        assert meta.split_version == 3 and meta.task_ids == s[split]
        assert [r.task_id for r in rows] == s[split]
        added, _ = load_run(str((meta.composed_of or [])[-1]))
        assert (
            meta.composed_of == [old, added.run_id]
            and added.task_ids == s[split][len(s["v2"][split]) :]
        )
        assert loop_run._runs_of(d, "v1", split, split_ids(d, split), 1)[0].run_id == run
    meta, results = load_run(BANKING_V1_V3[0])
    v1 = load_version(d, "v1")
    # v1's one run of split v3 and v2's, over all 60 train tasks; v2's test ids stay out
    record, _ = champion_record(v1, BANKING_V1_V3[0], loop_run._read_failures(d, results))
    assert "over the 60 train tasks you may read" in record
    assert "its gate compared its test run with v1's (test passes 6 → 5)" in record
    assert not [t for t in o["fixed"] + o["broken"] if t in record]
    [held] = held_challengers(d, "v1")
    assert held["reason"] == "the gate's test passes 6 → 5"
    assert held["passes"] == "2/60 → 17/60 on train" and held["fixed"] == o["read_fixed"]


@pytest.mark.skipif(not replay.available(), reason="tau2 is not installed")
def test_banking_s_sixty_train_tasks_fit_one_prompt_and_name_no_held_out_task() -> None:
    """With the gate on test (3 Oct 2026) the optimiser reads every train failure: v1's 58. Each
    failure's graded difference is cut to its share, and a test task a train task's notes name
    ("Adversarial variant of task_026") is never named."""
    import re

    from tau2_loop.data.splits import split_ids
    from tau2_loop.loop.optimiser import DIFF_BUDGET, _share, _unseen_ids, diff_block

    d = "banking_knowledge"
    test = split_ids(d, "test")
    assert _unseen_ids(d) == set(test) and _unseen_ids("airline") == set()
    meta, results = load_run(BANKING_V1_V3[0])
    failures = loop_run._read_failures(d, results)
    assert len(failures) == 58 and {r.task_id for r in failures} <= set(split_ids(d, "train"))
    share = _share(DIFF_BUDGET, len(failures))
    whole = diff_block(meta.run_id, next(r for r in failures if r.task_id == "task_041"), d)
    cut = diff_block(meta.run_id, next(r for r in failures if r.task_id == "task_041"), d, share)
    assert whole and cut and len(whole) > 20_000
    assert len(cut) < share + 80 and cut.endswith(
        "more lines cut, this failure's share of the prompt"
    )
    prompt = build_prompt(load_version(d, "v1"), "v9", meta.run_id, failures)
    assert (
        len(prompt) < 370_000
    )  # s14 added each task's required documents and the guide; s15 its lessons
    assert not [t for t in test if re.search(rf"\b{t}\b", prompt)]
    assert "Adversarial variant of a held-out task" in prompt
    assert "applies the gate there, against the champion's test run on the same tasks" in prompt
    assert "gate half" not in prompt


def test_a_banking_task_reads_as_prose_and_thirty_failures_share_one_budget() -> None:
    """Banking's scenario is one block of prose beside a persona (airline's is a structure):
    banking's first cycle crashed on it before any model call. Thirty failures split the
    transcript and scenario budgets; a handful (airline) keep full lengths."""
    from tau2_loop.loop.optimiser import (
        MIN_SHARE,
        SCENARIO_BUDGET,
        TRACE_CHARS,
        TRANSCRIPT_BUDGET,
        _calls,
        _share,
        task_block,
    )

    full = task_block("banking_knowledge", "task_079")
    assert "USER'S GOAL" in full and "You are Carlos Rodriguez" in full
    cut = task_block("banking_knowledge", "task_079", goal_chars=2000, actions=False)
    goal = next(ln for ln in cut.splitlines() if ln.startswith("USER'S GOAL"))
    assert len(cut) < len(full) and "…" in cut
    assert "EXPECTED ACTIONS: listed against the agent's calls under WHAT FAILED" in cut
    assert goal  # the cut keeps the opening of the goal
    assert _share(TRANSCRIPT_BUDGET, 6, TRACE_CHARS) == TRACE_CHARS  # airline: unchanged
    assert _share(TRANSCRIPT_BUDGET, 30, TRACE_CHARS) == TRANSCRIPT_BUDGET // 30
    assert _share(SCENARIO_BUDGET, 300) == MIN_SHARE
    # a tool called sixteen times is one entry, not sixteen
    pairs = [("call_discoverable_agent_tool", m) for m in range(40, 72, 2)]
    assert _calls(pairs, "msg") == (
        "call_discoverable_agent_tool at messages 40, 42, 44 … 70 (16 calls)"
    )
    assert _calls([("cancel_reservation", "39_10")], "id") == "39_10 cancel_reservation"


def test_optimise_freezes_new_agent_settings_and_shows_the_runs_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`make optimise EFFORT=high` (s14): the harness writes the new agent.yaml before the
    session and freezes it; the session reads the policy the champion's run had, not the
    extract's; diagnosis.json records the settings change; `.lavish/` is fenced."""
    import asyncio

    import tau2_loop.agent.versions as versions
    import tau2_loop.loop.optimiser as opt

    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(opt, "AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(opt, "RUNS_DIR", tmp_path / "runs")
    run = tmp_path / "runs" / "run0"
    run.mkdir(parents=True)
    (run / "tau2_results.json").write_text(
        json.dumps({"simulations": [{"policy": "THE VARIANT'S POLICY: use the shell tool."}]})
    )
    champ_dir = tmp_path / "agents" / "airline" / "v0"
    champ_dir.mkdir(parents=True)
    (champ_dir / "system.md").write_text("You are an agent.\n{policy}")
    (champ_dir / "agent.yaml").write_text("model: sonnet\neffort: medium\ntool_mode: native\n")
    champion = versions.load_version("airline", "v0")
    new_dir = tmp_path / "agents" / "airline" / "v1"
    prompts: list[str] = []
    sessions: list[Any] = []

    class FakeClient:
        def __init__(self, options: Any) -> None:
            sessions.append(options)

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            prompts.append(prompt)
            assert (new_dir / ".context" / "policy.md").read_text().startswith("THE VARIANT'S")
            assert "effort: high" in (new_dir / "agent.yaml").read_text()
            (new_dir / "system.md").write_text("You are a careful agent.\n{policy}")
            (new_dir / "diagnosis.json").write_text(json.dumps({"diagnoses": []}))

        async def receive_response(self) -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(opt, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(opt, "require_live", lambda: None)
    monkeypatch.setattr(opt, "subscription_env", lambda: {})

    out = asyncio.run(
        opt.run_optimiser(
            champion,
            "run0",
            [],
            mode="classic",
            settings={"effort": "high", "parallel_calls": True},
        )
    )
    assert not out.rejected and out.error is None, out.error
    v1 = versions.load_version("airline", "v1")
    assert v1.config.effort == "high" and v1.config.parallel_calls and v1.config.model == "sonnet"
    changes = ["effort: medium → high", "parallel_calls: False → True"]
    assert json.loads((new_dir / "diagnosis.json").read_text())["agent_yaml"] == changes
    assert "effort: medium → high" in prompts[0] and "as native tool calls" in prompts[0]
    guard_reads = sessions[0].hooks["PreToolUse"][1].hooks[0]
    page = asyncio.run(
        guard_reads({"tool_input": {"file_path": ".lavish/s13_plan.html"}}, None, None)
    )
    assert page["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert opt.bash_fence_reason("cat .lavish/s13_plan.html", []) is not None
    with pytest.raises(ValueError, match="change nothing"):
        asyncio.run(opt.run_optimiser(champion, "run0", [], settings={"effort": "medium"}))


def test_banking_task_blocks_name_the_required_documents_and_the_kb_is_copied(
    tmp_path: Path,
) -> None:
    import tau2_loop.loop.optimiser as opt
    from tau2_loop.data.splits import split_ids

    tid = next(t for t in split_ids("banking_knowledge", "train"))
    block = opt.task_block("banking_knowledge", tid)
    assert "REQUIRED DOCUMENTS" in block and "doc_" in block
    if opt._kb_source("banking_knowledge") is None:
        pytest.skip("no tau2 checkout")
    n = opt._copy_kb("banking_knowledge", tmp_path / "kb")
    files = sorted((tmp_path / "kb").glob("*.md"))
    assert n == len(files) > 600
    assert files[0].read_text().startswith("# ") and "ID: doc_" in files[0].read_text()
    assert opt._copy_kb("airline", tmp_path / "none") == 0
