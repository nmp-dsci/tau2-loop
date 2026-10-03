"""Offline units: splits, versions, summaries, the gate arithmetic, the billing guard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tau2_loop.agent.versions import list_versions, load_version, next_version_name
from tau2_loop.config import DOMAINS, SPLIT_SEED, SPLIT_VERSION, V1_SIZE
from tau2_loop.data.splits import read_split, read_task_extract, split_ids
from tau2_loop.eval.compare import compare, mcnemar_one_sided
from tau2_loop.eval.results import TaskResult, read_results, slug, summarise, write_results
from tau2_loop.llm import (
    BillingError,
    require_live,
    resolve_model,
    sdk_model,
    short_model,
    subscription_env,
)


def _r(tid: str, ok: bool, trial: int = 1) -> TaskResult:
    return TaskResult(task_id=tid, trial=trial, reward=1.0 if ok else 0.0, correct=ok)


# ── splits ───────────────────────────────────────────────────────────────
# (train, test, split version): half of each base set, except banking at split v3 (60 / 37)
SIDES = {
    "airline": (25, 25, SPLIT_VERSION),
    "retail": (57, 57, SPLIT_VERSION),
    "telecom": (57, 57, SPLIT_VERSION),
    "banking_knowledge": (60, 37, 3),
}


@pytest.mark.parametrize("domain", DOMAINS)
def test_committed_split_covers_the_base_set_disjoint_and_seeded(domain: str) -> None:
    s = read_split(domain)
    assert s["seed"] == SPLIT_SEED
    assert (len(s["train"]), len(s["test"]), s["version"]) == SIDES[domain]
    # every base task is train or test: nothing is held in reserve
    assert not set(s["train"]) & set(s["test"]) and "test_cap" not in s
    assert s["reserve_n"] == 0 and len(s["train"]) + len(s["test"]) == s["base_n"]
    assert split_ids(domain, "all") == s["train"] + s["test"]


def test_banking_split_v3_is_its_seeded_cut_and_moves_no_task_side() -> None:
    """The 24 tasks s09 held back are dealt, seed 300: 12 to train (60), 12 to test (37). Every
    v2 member keeps its side, so v1's runs (48 train, 25 test) are still on the sides they ran."""
    from tau2_loop.data.splits import cut

    s = read_split("banking_knowledge")
    assert cut("banking_knowledge") == s
    assert s["train"][:48] == s["v2"]["train"] and s["test"][:25] == s["v2"]["test"]
    assert set(s["v1"]["train"]) <= set(s["train"]) and set(s["v1"]["test"]) <= set(s["test"])
    read, gate = s["halves"]["read"], s["halves"]["gate"]
    assert (len(read), len(gate)) == (30, 30) and sorted(read + gate) == sorted(s["train"])
    # the halves keep their members and take six of the new train tasks each
    assert set(s["train"][48:]) == set(read[24:]) | set(gate[24:])


@pytest.mark.parametrize("domain", DOMAINS)
def test_split_v2_keeps_what_was_held_out_held_out(domain: str) -> None:
    """Every task a committed run scored on train is still train, and every test task still test."""
    from tau2_loop.eval.runner import list_runs

    s = read_split(domain)
    assert s["train"][:V1_SIZE] == s["v1"]["train"] and s["test"][:V1_SIZE] == s["v1"]["test"]
    for m in list_runs(domain):
        if m.split in ("train", "test") and not m.dry_run:
            assert set(m.task_ids) <= set(s[m.split]), m.run_id


@pytest.mark.parametrize("domain", DOMAINS)
def test_task_extract_covers_the_split(domain: str) -> None:
    ext = read_task_extract(domain)
    ids = {t["id"] for t in ext["tasks"]}
    held = set((read_split(domain).get("test_cap") or {}).get("held_back") or [])
    assert ids == set(split_ids(domain, "all")) | held
    assert ext["policy_words"] > 100
    assert ext["tools"]


def test_split_is_reproducible_from_seed() -> None:
    import random

    from tau2_loop.data.splits import _base_task_ids, cut_ids

    s = read_split("airline")
    base = _base_task_ids("airline")
    assert len(base) == s["base_n"]
    drawn = random.Random(SPLIT_SEED).sample(base, 2 * V1_SIZE)
    assert drawn[:V1_SIZE] == s["v1"]["train"] and drawn[V1_SIZE:] == s["v1"]["test"]
    c = cut_ids(base)
    assert c["train"] == s["train"] and c["test"] == s["test"]


def test_an_odd_reserve_puts_its_extra_task_on_the_reported_side() -> None:
    from tau2_loop.data.splits import cut_ids

    c = cut_ids([f"t{i}" for i in range(45)], v1_size=10)
    assert (len(c["train"]), len(c["test"])) == (22, 23)
    assert c["train"][:10] == c["v1"]["train"] and c["test"][:10] == c["v1"]["test"]


# ── versions ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("domain", DOMAINS + ("mock",))
def test_each_domain_has_its_base_version_with_policy_slot(domain: str) -> None:
    """v0 on Haiku, except banking, whose v0 was retired on 3 Oct 2026: v1, v0's prompt on Sonnet."""
    from tau2_loop.agent.versions import base_version

    name = base_version(domain)
    v = load_version(domain, name)
    assert "{policy}" in v.system_prompt
    model = "sonnet" if domain == "banking_knowledge" else "haiku"
    assert (name, v.config.model) == ("v1" if model == "sonnet" else "v0", model)
    assert v.config.tool_mode == "json" and len(v.fingerprint) == 12
    assert v.ref == f"{domain}/{name}"


def test_next_version_name_counts_per_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tau2_loop.agent.versions as versions

    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    assert next_version_name("airline") == "v0"
    (tmp_path / "airline" / "v3").mkdir(parents=True)
    (tmp_path / "airline" / "v3" / "system.md").write_text("x {policy}")
    assert [v.name for v in list_versions("airline")] == ["v3"]
    assert next_version_name("airline") == "v4"
    assert next_version_name("retail") == "v0"


# ── results ──────────────────────────────────────────────────────────────
def test_summarise_pass_hat_k() -> None:
    rows = [_r("a", True, 1), _r("a", True, 2), _r("b", True, 1), _r("b", False, 2)]
    s = summarise(rows)
    assert s.n_scored == 4 and s.passed == 3 and s.trials == 2
    assert s.pass_hat_k == {"pass^1": 0.75, "pass^2": 0.5}
    assert s.failed_ids == ["b"]


def test_results_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    rows = [_r("x", True), TaskResult("y", 1, 0.0, False, error="max_steps", purpose="p")]
    write_results(p, rows)
    assert read_results(p) == rows


def test_slug_is_filesystem_safe() -> None:
    s = slug("[mobile_data_issue]airplane_mode_on|user_abroad[PERSONA:Easy]")
    assert "/" not in s and "|" not in s and "[" not in s and ":" not in s
    assert len(slug("x" * 200)) <= 80


# ── gate ─────────────────────────────────────────────────────────────────
def test_mcnemar_values() -> None:
    assert mcnemar_one_sided(0, 0) == 1.0
    assert mcnemar_one_sided(5, 0) == pytest.approx(1 / 32)
    assert mcnemar_one_sided(4, 0) == pytest.approx(1 / 16)
    assert mcnemar_one_sided(3, 1) == pytest.approx(5 / 16)


def test_compare_pairs_by_task_and_trial() -> None:
    champ = [_r(str(i), i < 10) for i in range(20)]
    chall = [_r(str(i), i < 15) for i in range(20)]
    v = compare(champ, chall)
    assert v.fixed == ["10", "11", "12", "13", "14"] and v.broken == []
    assert v.promote and v.p_value == pytest.approx(1 / 32)
    assert v.champion_passed == 10 and v.challenger_passed == 15
    worse = compare(champ, [_r(str(i), i < 13 and i != 0) for i in range(20)])
    assert worse.broken == ["0"] and not worse.promote
    assert v.rule == "mcnemar"


def test_dominance_promotes_a_clean_gain_below_significance() -> None:
    """1 fixed / 0 broken is p = 0.5 — held by McNemar, promoted by dominance; 4/1 is neither."""
    champ = [_r(str(i), i < 12) for i in range(20)]
    clean1 = compare(champ, [_r(str(i), i < 13) for i in range(20)])
    assert clean1.promote and clean1.rule == "dominance" and clean1.p_value == pytest.approx(0.5)
    assert "dominance" in clean1.reason
    same = compare(champ, [_r(str(i), i < 12) for i in range(20)])
    assert not same.promote and same.rule == "none"
    four_one = compare(champ, [_r(str(i), (i < 16 and i != 0)) for i in range(20)])
    assert four_one.fixed == ["12", "13", "14", "15"] and four_one.broken == ["0"]
    assert not four_one.promote
    assert not compare(
        champ, [_r(str(i), i < 15) for i in range(20)], dominance_min_fixed=None
    ).promote


# ── billing ──────────────────────────────────────────────────────────────
def test_models_and_prefix() -> None:
    assert resolve_model("haiku") == "claude-haiku-4-5"
    assert sdk_model("haiku") == "claude-sdk/claude-haiku-4-5"
    assert resolve_model("claude-sdk/claude-haiku-4-5") == "claude-haiku-4-5"
    assert short_model("claude-sdk/claude-sonnet-5") == "sonnet"


def test_require_live_refuses_key_exported_by_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    import tau2_loop.config as cfg

    monkeypatch.setattr(cfg, "KEY_IN_SHELL_AT_START", True)
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    with pytest.raises(BillingError):
        require_live()


def test_require_live_scrubs_a_dotenv_injected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """tau2's load_dotenv() search can pull a key out of ~/.env; that one is dropped, not billed."""
    import tau2_loop.config as cfg

    monkeypatch.setattr(cfg, "KEY_IN_SHELL_AT_START", False)
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-dotenv")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    require_live()
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_require_live_refuses_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "1")
    with pytest.raises(BillingError):
        require_live()


def test_subscription_env_blanks_key_and_claude_code_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BILLING", "subscription")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("CLAUDECODE", "1")
    env = subscription_env()
    assert env["ANTHROPIC_API_KEY"] == "" and env["CLAUDECODE"] == ""
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test"  # the parent is untouched


def test_registry_and_ledger_are_per_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tau2_loop.config as cfg
    from tau2_loop.loop import ledger as led
    from tau2_loop.tracking import registry as reg

    monkeypatch.setattr(cfg, "LOOP_DIR", tmp_path)
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    monkeypatch.setattr(reg, "registry_path", lambda d: tmp_path / d / "registry.json")
    led.append_entry(
        "airline",
        {"cycle": 1, "champion": "v0", "challenger": "v1", "outcome": {"verdict": "pending"}},
    )
    led.update_entry("airline", 1, outcome={"verdict": "hold", "fixed": [], "broken": ["3"]})
    assert led.read_ledger("retail") == []
    assert led.read_ledger("airline")[0]["outcome"]["verdict"] == "hold"
    assert led.next_cycle_number("airline") == 2 and led.next_cycle_number("retail") == 1
    assert "BROKE" in led.prior_attempts("airline", "3")[0]["root_cause"]
    assert reg.read_registry("telecom")["champion"] is None
    assert json.loads(json.dumps(reg.read_all()))["airline"]["domain"] == "airline"


def test_tau2_sha_reads_the_actual_submodule_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    from tau2_loop.eval import runner

    assert runner._tau2_sha() == runner.TAU2_SHA_FALLBACK  # the real submodule is pinned here

    monkeypatch.setattr(runner, "ROOT", Path("/nonexistent-path-for-test"))
    assert runner._tau2_sha() == runner.TAU2_SHA_FALLBACK  # git fails -> fallback, not a crash

    meta = runner.RunMeta(
        "r", "airline", "v0", "fp", "m", "u", "j", "train", 20, 1, 3, 300, "t", "t"
    )
    assert meta.tau2_sha == runner.TAU2_SHA_FALLBACK


def test_seconds_until_reset_parses_the_cli_message() -> None:
    from datetime import datetime

    from tau2_loop.llm.sdk_provider import seconds_until_reset

    msg = "ResultError: You've hit your session limit · resets 4:40pm (Australia/Sydney) (exit code: 1)"
    now = datetime(2026, 9, 15, 14, 0)
    assert seconds_until_reset(msg, now) == (2 * 3600 + 40 * 60) + 60
    assert seconds_until_reset(msg, datetime(2026, 9, 15, 17, 0)) == 6 * 3600  # capped: tomorrow
    assert (
        seconds_until_reset("resets 12am", datetime(2026, 9, 15, 23, 0)) is None
    )  # not a limit msg
    assert seconds_until_reset("session limit reached", now) == 15 * 60
    assert seconds_until_reset("ProcessError: boom", now) is None


def test_redaction_removes_the_account_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI names its account in every session; nothing we write may carry the address."""
    import tau2_loop.llm as llm

    monkeypatch.setattr(llm, "account_email", lambda: "someone@example.com")
    assert llm.redact("email someone@example.com please") == f"email {llm.REDACTED_EMAIL} please"
    (tmp_path / "a.json").write_text('{"x": "someone@example.com"}')
    (tmp_path / "b.txt").write_text("clean")
    assert llm.redact_tree(tmp_path) == 1
    assert "someone" not in (tmp_path / "a.json").read_text()
    monkeypatch.setattr(llm, "account_email", lambda: "")
    assert llm.redact("keep@example.com") == "keep@example.com"


def test_ledger_entries_are_redacted_on_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnosis/transcript text can quote the account email; the ledger must never store it."""
    import tau2_loop.llm as llm
    from tau2_loop.loop import ledger as led

    monkeypatch.setattr(llm, "account_email", lambda: "someone@example.com")
    monkeypatch.setattr(led, "ledger_path", lambda d: tmp_path / d / "ledger.jsonl")
    led.append_entry(
        "airline",
        {
            "cycle": 1,
            "diagnoses": "customer emailed someone@example.com about a refund",
        },
    )
    raw = (tmp_path / "airline" / "ledger.jsonl").read_text()
    assert "someone@example.com" not in raw
    assert llm.REDACTED_EMAIL in raw

    led.update_entry("airline", 1, prompt_diff_summary="cc someone@example.com")
    entries = led.read_ledger("airline")
    assert "someone@example.com" not in json.dumps(entries)
    assert entries[0]["prompt_diff_summary"] == f"cc {llm.REDACTED_EMAIL}"


# ── the gate with trials, and the refusals ─────────────────────────────────
def test_compare_scores_a_task_by_its_pass_fraction_over_trials() -> None:
    def rows(fracs: dict[str, int], trials: int = 3) -> list[TaskResult]:
        return [_r(t, i < k, trial=i + 1) for t, k in fracs.items() for i in range(trials)]

    champ = rows({"a": 1, "b": 3, "c": 0, "d": 2})
    chall = rows({"a": 2, "b": 3, "c": 0, "d": 1})
    v = compare(champ, chall)
    assert v.trials == 3 and v.n == 4
    assert v.fixed == ["a"] and v.broken == ["d"]  # ⅓ → ⅔ and ⅔ → ⅓; trial order is irrelevant
    assert v.champion_passed == 6 and v.challenger_passed == 6
    assert v.champion_pass1 == v.challenger_pass1 == 0.5
    assert not v.promote and "3 trials" in v.reason


def test_compare_refuses_runs_over_different_tasks_or_trials() -> None:
    champ = [_r(str(i), i < 10) for i in range(20)]
    with pytest.raises(ValueError, match="different tasks"):
        compare(champ, [_r(str(i), i < 10) for i in range(25)])
    with pytest.raises(ValueError, match="different trials"):
        compare(champ, [_r(str(i), True, trial=t) for i in range(20) for t in (1, 2)])


def test_check_counts_on_a_committed_run() -> None:
    from tau2_loop.eval.results import check_counts
    from tau2_loop.eval.runner import load_run

    _, results = load_run("20260915T132148Z_airline_v2_train")
    c = check_counts(results)
    assert (c["db"]["passed"], c["db"]["n"], c["db"]["scored"]) == (16, 20, 20)
    assert (c["actions"]["passed"], c["actions"]["n"]) == (12, 17)
    assert (c["actions"]["items_met"], c["actions"]["items"], c["actions"]["scored"]) == (45, 55, 0)
    assert (c["communicate"]["passed"], c["communicate"]["n"]) == (2, 3)
    assert "env" not in c and "nl" not in c


# ── the fork ────────────────────────────────────────────────────────────
def test_fork_copies_the_surfaces_and_records_the_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    import tau2_loop.agent.versions as versions

    real = versions.AGENTS_DIR
    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    shutil.copytree(real / "airline" / "v0", tmp_path / "airline" / "v0")
    src = versions.load_version("airline", "v0")
    v = versions.fork_version("airline", "v0", model="sonnet")
    assert v.name == "v1" and v.config.model == "sonnet" and v.config.effort == "medium"
    assert v.system_prompt == src.system_prompt and v.helper == src.helper
    assert (
        v.fingerprint != src.fingerprint
    )  # agent.yaml differs, so the bytes a run is logged on do
    d = json.loads((v.path / "diagnosis.json").read_text())
    assert d["kind"] == "model swap" and d["forked_from"] == "v0"
    assert versions.lineage("airline", "v1") == ["v1", "v0"]
    with pytest.raises(ValueError):
        versions.fork_version("airline", "v1", model="sonnet")  # nothing would change
