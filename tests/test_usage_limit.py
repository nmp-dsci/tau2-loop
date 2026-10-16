"""s25 (the person's call, 15 Oct 2026): a weekly usage limit voids a run instead of scoring it.

v25's 34 conversations each met "You've hit your weekly limit" and were scored as failures, so the
run read 0 of 17. The core now fails every call after a weekly limit at once, and a conversation
that ends in an infrastructure error is left unscored. No model is called.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tau2_loop.eval import results
from tau2_loop.llm import core

WEEKLY = "ResultError: You've hit your weekly limit · resets Oct 15 at 12pm (Australia/Sydney)"
SESSION = "ResultError: You've hit your session limit · resets 4:40pm (Australia/Sydney)"


def test_a_weekly_limit_is_not_waited_for_and_every_later_call_fails_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "_hard_limit", None)
    monkeypatch.setattr(core, "check_billing", lambda: None)
    sleeps: list[float] = []
    monkeypatch.setattr(core.time, "sleep", sleeps.append)
    calls: list[str] = []

    async def fake_query(*args: Any, **kwargs: Any) -> core.SdkResult:
        calls.append("q")
        return core.SdkResult("", 0, 0, None, 0, None, error=WEEKLY)

    monkeypatch.setattr(core, "_query", fake_query)
    first = core.run_query("sys", "hi", "claude-sonnet-5-5")
    assert first.error == WEEKLY and calls == ["q"] and sleeps == []  # no retry, no wait
    second = core.run_query("sys", "again", "claude-sonnet-5-5")
    assert second.error == WEEKLY and calls == ["q"]  # never reached the SDK


def test_a_reset_lets_the_next_call_reach_the_sdk_and_a_new_run_start_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "_hard_limit", None)
    monkeypatch.setattr(core, "check_billing", lambda: None)
    calls: list[str] = []

    async def fake_query(*args: Any, **kwargs: Any) -> core.SdkResult:
        calls.append("q")
        return core.SdkResult("", 0, 0, None, 0, None, error=WEEKLY)

    monkeypatch.setattr(core, "_query", fake_query)
    core.run_query("sys", "hi", "claude-sonnet-5-5")
    core.run_query("sys", "again", "claude-sonnet-5-5")
    assert calls == ["q"]
    core.reset_hard_limit()
    core.run_query("sys", "after reset", "claude-sonnet-5-5")
    assert calls == ["q", "q"]

    from tau2_loop import llm
    from tau2_loop.eval.runner import run_eval

    class StopError(Exception):
        pass

    def stop() -> None:
        raise StopError

    monkeypatch.setattr(llm, "require_live", stop)
    core.run_query("sys", "again", "claude-sonnet-5-5")
    assert core._hard_limit == WEEKLY
    with pytest.raises(StopError):
        run_eval("airline")
    assert core._hard_limit is None


def test_only_the_session_limit_is_waited_for() -> None:
    assert core.hard_limit(WEEKLY)
    assert not core.hard_limit(SESSION)
    assert core.seconds_until_reset(WEEKLY) is None
    assert core.seconds_until_reset(SESSION) is not None
    assert not core.hard_limit("ProcessError: boom")


def _sim(reason: str, reward: float | None) -> Any:
    ri = (
        None
        if reward is None
        else SimpleNamespace(
            reward=reward,
            reward_basis=[],
            db_check=None,
            env_assertions=[],
            action_checks=[],
            communicate_checks=[],
            nl_assertions=[],
        )
    )
    return SimpleNamespace(
        task_id="task_001",
        trial=0,
        reward_info=ri,
        messages=[],
        termination_reason=reason,
        duration=1.0,
        info={"error": WEEKLY} if reason == "infrastructure_error" else {},
    )


def test_a_conversation_lost_to_the_harness_is_never_scored() -> None:
    task = SimpleNamespace(description=None)
    void = results.from_simulation(_sim("infrastructure_error", 0.0), task, "t.json")
    failed = results.from_simulation(_sim("user_stop", 0.0), task, "t.json")
    passed = results.from_simulation(_sim("user_stop", 1.0), task, "t.json")
    assert void.correct is None and void.error and "weekly limit" in void.error
    assert failed.correct is False and passed.correct is True
    s = results.summarise([void, failed, passed])
    assert (s.n, s.n_scored, s.passed) == (3, 2, 1)
    assert s.pass_rate == 0.5 and s.errored_ids == ["task_001"]
    v = results.summarise([void, void])
    assert v.n_scored == 0 and v.pass_rate is None  # a void run is not "0 of 17"
