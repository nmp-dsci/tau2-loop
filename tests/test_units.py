"""Offline units: splits, versions, summaries, the gate arithmetic, the billing guard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tau2_loop.agent.versions import list_versions, load_version, next_version_name
from tau2_loop.config import DOMAINS, SPLIT_SEED, SPLIT_SIZE
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
@pytest.mark.parametrize("domain", DOMAINS)
def test_committed_split_is_20_20_disjoint_and_seeded(domain: str) -> None:
    s = read_split(domain)
    assert s["seed"] == SPLIT_SEED and s["size"] == SPLIT_SIZE
    assert len(s["train"]) == 20 and len(s["test"]) == 20
    assert not set(s["train"]) & set(s["test"])
    assert s["base_n"] >= 40 and s["reserve_n"] == s["base_n"] - 40
    assert split_ids(domain, "all") == s["train"] + s["test"]


@pytest.mark.parametrize("domain", DOMAINS)
def test_task_extract_covers_the_split(domain: str) -> None:
    ext = read_task_extract(domain)
    ids = {t["id"] for t in ext["tasks"]}
    assert ids == set(split_ids(domain, "all"))
    assert ext["policy_words"] > 100
    assert ext["tools"]


def test_split_is_reproducible_from_seed() -> None:
    import random

    from tau2_loop.data.splits import _base_task_ids

    s = read_split("airline")
    base = _base_task_ids("airline")
    assert len(base) == s["base_n"]
    drawn = random.Random(SPLIT_SEED).sample(base, 40)
    assert drawn[:20] == s["train"] and drawn[20:] == s["test"]


# ── versions ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("domain", DOMAINS + ("mock",))
def test_v0_exists_with_policy_slot(domain: str) -> None:
    v = load_version(domain, "v0")
    assert "{policy}" in v.system_prompt
    assert v.config.model == "haiku" and v.config.tool_mode == "json"
    assert len(v.fingerprint) == 12
    assert v.ref == f"{domain}/v0"


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
