"""The promotion gate: a one-sided exact sign test that the challenger beats the champion.

Two runs on the same tasks (same domain, same split, same trials) are paired by
task. A task's score under a run is its pass fraction over its trials (with
one trial, pass or fail); `b` tasks whose fraction rose are the ones the
challenger fixed, `c` tasks whose fraction fell the ones it broke, and the rest
carry no information. Under the null each changed task is a coin flip, so the
one-sided p-value is P(X ≤ c | n = b + c, p = ½) — McNemar's exact test when
trials = 1. Promote when p < alpha. The test is blunt by construction — five
fixes and no breaks is the smallest result that clears 0.05 (p = 1/32), seven
with one break — so the verdict carries b, c and p, not just a word. Pairing
trial 2 with trial 2 would treat two independent samples as a pair; a fraction
per task does not. Runs over different task sets or trial counts are refused,
not intersected: a comparison on the overlap would hide what was left out.

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

from tau2_loop.eval.results import TaskResult

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
    n: int  # tasks
    fixed: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    p_value: float = 1.0
    alpha: float = ALPHA
    reason: str = ""
    rule: str = "none"  # "mcnemar" | "dominance" | "none"
    trials: int = 1
    champion_pass1: float = 0.0  # mean pass fraction over the tasks
    challenger_pass1: float = 0.0


def pass_fractions(results: list[TaskResult]) -> dict[str, tuple[int, int]]:
    """task id → (passes, scored trials); a task whose every row is unscored reads (0, 0)."""
    out: dict[str, tuple[int, int]] = {}
    for r in results:
        p, n = out.get(r.task_id, (0, 0))
        if r.correct is not None:
            p, n = p + int(bool(r.correct)), n + 1
        out[r.task_id] = (p, n)
    return out


def _frac(pn: tuple[int, int]) -> float:
    return pn[0] / pn[1] if pn[1] else 0.0


def compare(
    champion: list[TaskResult],
    challenger: list[TaskResult],
    alpha: float = ALPHA,
    dominance_min_fixed: int | None = DOMINANCE_MIN_FIXED,
) -> Verdict:
    a, b = pass_fractions(champion), pass_fractions(challenger)
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise ValueError(
            f"the runs cover different tasks: {len(only_a)} only in the champion's "
            f"{only_a[:3]}, {len(only_b)} only in the challenger's {only_b[:3]}"
        )
    trials_a = {len([r for r in champion if r.task_id == t]) for t in a}
    trials_b = {len([r for r in challenger if r.task_id == t]) for t in b}
    if trials_a != trials_b or len(trials_a) > 1:
        raise ValueError(
            f"the runs carry different trials per task: {sorted(trials_a)} vs {sorted(trials_b)}"
        )
    tasks = sorted(a)
    fixed = [t for t in tasks if _frac(b[t]) > _frac(a[t])]
    broken = [t for t in tasks if _frac(b[t]) < _frac(a[t])]
    p = mcnemar_one_sided(len(fixed), len(broken))
    significant = p < alpha
    dominant = dominance_min_fixed is not None and not broken and len(fixed) >= dominance_min_fixed
    promote = significant or dominant
    rule = "mcnemar" if significant else "dominance" if dominant else "none"
    trials = next(iter(trials_a), 1)
    test_name = "McNemar one-sided" if trials == 1 else f"sign test one-sided ({trials} trials)"
    reason = (
        f"{test_name}: fixed {len(fixed)}, broke {len(broken)}, p = {p:.3f} "
        f"{'<' if significant else '≥'} α = {alpha}"
    )
    if dominant and not significant:
        reason += f" · promoted by dominance: broke 0, fixed ≥ {dominance_min_fixed}"
    n = len(tasks)
    return Verdict(
        promote=promote,
        champion_passed=sum(x[0] for x in a.values()),
        challenger_passed=sum(x[0] for x in b.values()),
        n=n,
        fixed=fixed,
        broken=broken,
        p_value=p,
        alpha=alpha,
        reason=reason,
        rule=rule,
        trials=trials,
        champion_pass1=round(sum(_frac(a[t]) for t in tasks) / n, 4) if n else 0.0,
        challenger_pass1=round(sum(_frac(b[t]) for t in tasks) / n, 4) if n else 0.0,
    )
