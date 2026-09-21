"""Log a run folder to the central MLflow (nmp-central-ai, http://localhost:5000).

MLflow is the index, never the record: every artifact it holds is a copy of a
file in `runs/<id>/`. If the server is down the eval still completes and the
run folder is complete; `make snapshot` exports the experiment for the demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from tau2_loop.config import settings

if TYPE_CHECKING:
    from tau2_loop.eval.results import TaskResult
    from tau2_loop.eval.runner import RunMeta

EXPERIMENT = "tau2-loop"


def _client_setup() -> None:
    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)


def log_run(run_dir: Path, meta: RunMeta, results: list[TaskResult]) -> str:
    """One MLflow run per eval run; returns the MLflow run id."""
    _client_setup()
    s = meta.summary or {}
    with mlflow.start_run(run_name=meta.run_id) as run:
        mlflow.set_tags(
            {
                "domain": meta.domain,
                "agent": meta.agent,
                "fingerprint": meta.fingerprint,
                "model": meta.model,
                "split": meta.split,
                "code_sha": meta.code_sha,
                "kind": "eval",
            }
        )
        mlflow.log_params(
            {
                "domain": meta.domain,
                "agent": meta.agent,
                "model": meta.model,
                "user_model": meta.user_model,
                "split": meta.split,
                "trials": meta.trials,
                "concurrency": meta.concurrency,
                "n_tasks": meta.n_tasks,
                "fingerprint": meta.fingerprint,
                "tool_mode": meta.tool_mode,
            }
        )
        if s.get("n_scored"):
            mlflow.log_metrics(
                {
                    "passed": float(s["passed"]),
                    "pass_rate": float(s["pass_rate"] or 0.0),
                    **{
                        k.replace("^", "_hat_"): float(v)
                        for k, v in (s.get("pass_hat_k") or {}).items()
                    },
                }
            )
        mlflow.log_metrics(
            {
                "cost_usd_est": float(s.get("cost_usd_est", 0.0)),
                "duration_s": float(s.get("duration_ms", 0)) / 1000,
                "errors": float(len(s.get("errored_ids", []))),
                "agent_turns_total": float(sum(r.n_agent_turns for r in results)),
                "tool_calls_total": float(sum(r.n_tool_calls for r in results)),
                "agent_input_tokens": float(sum(r.agent_input_tokens for r in results)),
                "agent_output_tokens": float(sum(r.agent_output_tokens for r in results)),
                "user_input_tokens": float(sum(r.user_input_tokens for r in results)),
                "user_output_tokens": float(sum(r.user_output_tokens for r in results)),
            }
        )
        for r in results:
            if r.correct is not None:
                key = f"task_{r.task_id}_t{r.trial}"[:250]
                key = "".join(ch if ch.isalnum() or ch in "_-./ " else "_" for ch in key)
                mlflow.log_metric(key, 1.0 if r.correct else 0.0)
        for name in ("run.json", "results.jsonl"):
            if (run_dir / name).exists():
                mlflow.log_artifact(str(run_dir / name))
        if (run_dir / "agent").is_dir():
            mlflow.log_artifacts(str(run_dir / "agent"), artifact_path="agent")
        if (run_dir / "traces").is_dir():
            mlflow.log_artifacts(str(run_dir / "traces"), artifact_path="traces")
        return str(run.info.run_id)


def log_cycle(entry: dict[str, object]) -> str | None:
    """A loop cycle as its own MLflow run, tagged `kind=cycle`, so the UI shows the story."""
    try:
        _client_setup()
        with mlflow.start_run(run_name=f"{entry.get('domain')}-cycle-{entry.get('cycle')}") as run:
            mlflow.set_tags(
                {
                    "kind": "cycle",
                    "domain": str(entry.get("domain")),
                    "champion": str(entry.get("champion")),
                    "challenger": str(entry.get("challenger")),
                }
            )
            outcome = entry.get("outcome") or {}
            if isinstance(outcome, dict):
                mlflow.set_tag("verdict", str(outcome.get("verdict")))
            tokens = entry.get("tokens") or {}
            if isinstance(tokens, dict):
                mlflow.log_metrics(
                    {
                        f"tokens_{k}": float(v)
                        for k, v in tokens.items()
                        if isinstance(v, int | float)
                    }
                )
            mlflow.log_dict(dict(entry), "ledger_entry.json")
            return str(run.info.run_id)
    except Exception:  # noqa: BLE001
        return None
