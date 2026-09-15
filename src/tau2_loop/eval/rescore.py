"""Re-score a run folder offline from its saved trajectories.

`tau2_results.json` holds every conversation. Replaying it through tau2's
evaluators recomputes the DB hash, the env assertions, the action checks and
the communicate checks with no model in the loop; only the NL-assertion judge
(retail's basis) needs an LLM, and for that component the recorded verdict is
reused. The reward is then the product over the task's `reward_basis`, exactly
as `evaluate_simulation(ALL)` forms it, so the CI gate can check that a
committed champion still scores what its registry entry says.
"""

from __future__ import annotations

import json
from typing import Any

from tau2_loop.config import RUNS_DIR, quiet_tau2


def rescore_run(run_id: str) -> dict[str, Any]:
    """{task_id#trial: {recorded, rescored, components}} for every simulation in the run."""
    quiet_tau2()
    from tau2.data_model.simulation import Results
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode

    p = RUNS_DIR / run_id / "tau2_results.json"
    if not p.exists():
        raise FileNotFoundError(f"{run_id}: no tau2_results.json to re-score")
    results = Results.model_validate(json.loads(p.read_text()))
    domain = results.info.environment_info.domain_name
    env_kwargs: dict[str, Any] = (
        {"retrieval_variant": "bm25"} if domain == "banking_knowledge" else {}
    )
    tasks = {t.id: t for t in results.tasks}
    out: dict[str, Any] = {}
    for sim in results.simulations:
        task = tasks[sim.task_id]
        recorded = float(sim.reward_info.reward) if sim.reward_info else 0.0
        basis = set(task.evaluation_criteria.reward_basis) if task.evaluation_criteria else set()
        components: dict[str, float] = {}
        reward = 1.0
        if sim.termination_reason.value not in {"agent_stop", "user_stop"}:
            reward = 0.0
            components["terminated"] = 0.0
        elif task.evaluation_criteria is None:
            reward = 1.0
        else:
            parts = [
                ({RewardType.DB, RewardType.ENV_ASSERTION}, EvaluationType.ENV, "env"),
                ({RewardType.ACTION}, EvaluationType.ACTION, "action"),
                ({RewardType.COMMUNICATE}, EvaluationType.COMMUNICATE, "communicate"),
            ]
            for kinds, etype, label in parts:
                if basis & kinds:
                    ri = evaluate_simulation(
                        simulation=sim,
                        task=task,
                        evaluation_type=etype,
                        solo_mode=False,
                        domain=domain,
                        mode=CommunicationMode.HALF_DUPLEX,
                        env_kwargs=env_kwargs,
                        strict_replay=False,
                    )
                    components[label] = float(ri.reward)
                    reward *= float(ri.reward)
            if RewardType.NL_ASSERTION in basis:
                nl = sim.reward_info.nl_assertions if sim.reward_info else None
                nl_reward = 1.0 if nl and all(a.met for a in nl) else 0.0
                components["nl_assertions(recorded)"] = nl_reward
                reward *= nl_reward
        key = f"{sim.task_id}#t{int(sim.trial or 0) + 1}"
        out[key] = {
            "recorded": recorded,
            "rescored": round(reward, 4),
            "components": components,
            "agree": (recorded >= 1.0) == (reward >= 1.0),
        }
    return out
