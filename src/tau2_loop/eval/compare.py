"""The promotion gate: a one-sided exact McNemar test that the challenger beats the champion.

Two runs on the same tasks (same domain, same split, same seed) are paired by
task id and trial. Only the discordant pairs carry information: `b` tasks the
challenger fixed, `c` tasks it broke. Under the null each discordant task is a
coin flip, so the one-sided p-value is P(X ≤ c | n = b + c, p = ½). Promote
when p < alpha. With twenty tasks the test is blunt by construction — five
fixes and no breaks is the smallest result that clears 0.05 (p = 1/32) — so
the verdict carries b, c and p, not just a word.

Second path, decided at the s01 review: a challenger that breaks nothing has
no observed downside, so it is also promoted when it fixes at least one task
and breaks none ("dominance", DOMINANCE_MIN_FIXED = 1). That is a decision
rule, not a significance test — 1 / 0 is p = 0.5, a coin flip, and twenty
clean tasks are consistent with an unobserved regression rate of ~14%. The
held-out test run that follows every promotion is what keeps it honest, and
the ledger records which rule fired so the two kinds never blur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

from tau2_loop.eval.results import TaskResult, summarise

ALPHA = 0.05
DOMINANCE_MIN_FIXED = 1  # promote on fixed ≥ this and broke == 0, whatever p says


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
    rule: str = "none"  # "mcnemar" | "dominance" | "none"


def _key(r: TaskResult) -> str:
    return r.task_id if r.trial in (0, 1) else f"{r.task_id}#t{r.trial}"


def compare(
    champion: list[TaskResult],
    challenger: list[TaskResult],
    alpha: float = ALPHA,
    dominance_min_fixed: int | None = DOMINANCE_MIN_FIXED,
) -> Verdict:
    a = {_key(r): r for r in champion if r.correct is not None}
    b_ = {_key(r): r for r in challenger if r.correct is not None}
    common = sorted(set(a) & set(b_))
    fixed = [t for t in common if not a[t].correct and b_[t].correct]
    broken = [t for t in common if a[t].correct and not b_[t].correct]
    sa, sb = summarise([a[t] for t in common]), summarise([b_[t] for t in common])
    p = mcnemar_one_sided(len(fixed), len(broken))
    significant = p < alpha
    dominant = dominance_min_fixed is not None and not broken and len(fixed) >= dominance_min_fixed
    promote = significant or dominant
    rule = "mcnemar" if significant else "dominance" if dominant else "none"
    reason = (
        f"McNemar one-sided: fixed {len(fixed)}, broke {len(broken)}, p = {p:.3f} "
        f"{'<' if significant else '≥'} α = {alpha}"
    )
    if dominant and not significant:
        reason += f" · promoted by dominance: broke 0, fixed ≥ {dominance_min_fixed}"
    return Verdict(
        promote, sa.passed, sb.passed, len(common), fixed, broken, p, alpha, reason, rule
    )
