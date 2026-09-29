"""What a run cost, as a distribution rather than a total.

`summarise()` reports sums and two means, which hide the tail that actually
decides whether a full-split run fits in a subscription window: one conversation
in forty runs to the turn cap and spends ten times the median. This gives every
metric its mean, median, p95 and max beside the sum, plus the two ratios worth
reading — cost per passing conversation and per conversation attempted.

Nothing here reaches a model or a tracking server; it is arithmetic over the
rows already committed in `runs/<id>/results.jsonl`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tau2_loop.eval.results import TaskResult

# The metrics a conversation is measured by, in the order a table shows them.
METRICS: tuple[str, ...] = (
    "turns",
    "tool_calls",
    "tool_errors",
    "messages",
    "wall_s",
    "agent_in",
    "agent_out",
    "user_in",
    "user_out",
    "tokens",
    "cost_usd",
)


@dataclass(frozen=True)
class Stat:
    mean: float
    p50: float
    p95: float
    max: float
    sum: float


def _pct(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile: with 20 conversations, p95 is the 19th, not an
    interpolation between two of them. Blunt on purpose and easy to check by eye."""
    if not sorted_values:
        return 0.0
    i = min(len(sorted_values) - 1, max(0, round(q * len(sorted_values) + 0.5) - 1))
    return sorted_values[i]


def _stat(values: list[float]) -> Stat:
    if not values:
        return Stat(0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)
    total = sum(ordered)
    return Stat(
        mean=round(total / len(ordered), 4),
        p50=round(_pct(ordered, 0.5), 4),
        p95=round(_pct(ordered, 0.95), 4),
        max=round(ordered[-1], 4),
        sum=round(total, 4),
    )


def _values(results: list[TaskResult], metric: str) -> list[float]:
    match metric:
        case "turns":
            return [float(r.n_agent_turns) for r in results]
        case "tool_calls":
            return [float(r.n_tool_calls) for r in results]
        case "tool_errors":
            return [float(r.n_tool_errors) for r in results]
        case "messages":
            return [float(r.n_messages) for r in results]
        case "wall_s":
            return [r.duration_ms / 1000 for r in results]
        case "agent_in":
            return [float(r.agent_input_tokens) for r in results]
        case "agent_out":
            return [float(r.agent_output_tokens) for r in results]
        case "user_in":
            return [float(r.user_input_tokens) for r in results]
        case "user_out":
            return [float(r.user_output_tokens) for r in results]
        case "tokens":
            return [
                float(
                    r.agent_input_tokens
                    + r.agent_output_tokens
                    + r.user_input_tokens
                    + r.user_output_tokens
                )
                for r in results
            ]
        case "cost_usd":
            return [float(r.cost_usd_est or 0.0) for r in results]
    raise KeyError(metric)


def profile(results: list[TaskResult]) -> dict[str, Any]:
    """The run's distribution. `n` is conversations, not tasks: a 4-trial run of
    20 tasks profiles 80 of them."""
    n = len(results)
    scored = [r for r in results if r.correct is not None]
    passed = [r for r in scored if r.correct]
    cost = sum(r.cost_usd_est or 0.0 for r in results)
    tokens = sum(_values(results, "tokens"))
    capped = sum(1 for r in results if r.termination_reason == "max_steps")
    return {
        "n": n,
        "metrics": {m: asdict(_stat(_values(results, m))) for m in METRICS},
        "cost_per_pass": round(cost / len(passed), 4) if passed and cost else None,
        "cost_per_conversation": round(cost / n, 4) if n and cost else None,
        "tokens_per_pass": round(tokens / len(passed)) if passed and tokens else None,
        "error_rate": round(sum(1 for r in results if r.error) / n, 4) if n else None,
        "fail_rate": round(sum(1 for r in scored if not r.correct) / len(scored), 4)
        if scored
        else None,
        # a conversation that ran out of turns never got to answer: a different
        # failure from a wrong answer, and the one that a turn cap can fix
        "hit_turn_cap": capped,
        "hit_turn_cap_rate": round(capped / n, 4) if n else None,
    }
