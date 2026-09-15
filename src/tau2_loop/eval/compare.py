"""The promotion gate: a one-sided exact McNemar test that the challenger beats the champion.

Two runs on the same tasks (same domain, same split, same seed) are paired by
task id and trial. Only the discordant pairs carry information: `b` tasks the
challenger fixed, `c` tasks it broke. Under the null each discordant task is a
coin flip, so the one-sided p-value is P(X ≤ c | n = b + c, p = ½). Promote
when p < alpha. With twenty tasks the test is blunt by construction — five
fixes and no breaks is the smallest result that clears 0.05 (p = 1/32) — so
the verdict carries b, c and p, not just a word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

from tau2_loop.eval.results import TaskResult, summarise

ALPHA = 0.05


def mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided p-value that the challenger is better: P(breaks ≤ c | n = b + c, ½)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(c + 1))) / float(2**n)


@dataclass
class Verdict:
    promote: bool
    champion_passed: int
    challenger_passed: int
    n: int
    fixed: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    p_value: float = 1.0
    alpha: float = ALPHA
    reason: str = ""


def _key(r: TaskResult) -> str:
    return r.task_id if r.trial in (0, 1) else f"{r.task_id}#t{r.trial}"


def compare(
    champion: list[TaskResult], challenger: list[TaskResult], alpha: float = ALPHA
) -> Verdict:
    a = {_key(r): r for r in champion if r.correct is not None}
    b_ = {_key(r): r for r in challenger if r.correct is not None}
    common = sorted(set(a) & set(b_))
    fixed = [t for t in common if not a[t].correct and b_[t].correct]
    broken = [t for t in common if a[t].correct and not b_[t].correct]
    sa, sb = summarise([a[t] for t in common]), summarise([b_[t] for t in common])
    p = mcnemar_one_sided(len(fixed), len(broken))
    promote = p < alpha
    reason = (
        f"McNemar one-sided: fixed {len(fixed)}, broke {len(broken)}, p = {p:.3f} "
        f"{'<' if promote else '≥'} α = {alpha}"
    )
    return Verdict(promote, sa.passed, sb.passed, len(common), fixed, broken, p, alpha, reason)
