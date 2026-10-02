"""The tool judge in the central MLflow: experiment `tau2-loop/judge`, an index of committed files.

J1's golden answers are one run tagged `kind=judge-gold` (the gold file and its summary as
artifacts) and the evaluation dataset `tau2-loop.<domain>.golden`; J2's replays are runs tagged
`kind=judge-replay`, with the judge model as `plan_judge_model` (`judge_model` is tau2's NL
judge in `run.json`). Metrics and a verdict table, never traces (platform rule 4). The files stay
the truth: a failure here is printed, never raised, and nothing is lost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tau2_loop.tracking.mlflow_log import PROJECT, required_tags

EXPERIMENT = f"{PROJECT}/judge"


def dataset_name(domain: str) -> str:
    return f"{PROJECT}.{domain}.golden"


def _experiment() -> str:
    import mlflow

    from tau2_loop.config import settings

    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    exp = mlflow.set_experiment(EXPERIMENT)
    return str(exp.experiment_id)


def log_gold(domain: str) -> str:
    """The golden answers as an MLflow run and evaluation dataset; returns a one-line receipt."""
    import mlflow

    from tau2_loop.tooljudge import gold

    try:
        exp_id = _experiment()
        s = gold.read_summary(domain) or gold.write_summary(domain)
        with mlflow.start_run(run_name=f"{domain}-golden") as run:
            mlflow.set_tags({**required_tags(), "kind": "judge-gold", "domain": domain})
            mlflow.log_params(
                {"annotator_model": gold.ANNOTATOR_MODEL, "annotator_effort": gold.ANNOTATOR_EFFORT}
            )
            mlflow.log_metrics(
                {
                    "records": float(s["records"]),
                    "machine_checks_pass": float(s["machine_checks"]["pass"]),
                    "review_items": float(s["review_queue"]["items"]),
                    **{f"tokens_{k}": float(v) for k, v in (s.get("tokens") or {}).items()},
                }
            )
            for p in (gold.gold_path(domain), gold.summary_path(domain)):
                mlflow.log_artifact(str(p))
            run_id = run.info.run_id
        n = _dataset(domain, exp_id)
        return f"run {run_id} in {EXPERIMENT}; dataset {dataset_name(domain)} with {n} records"
    except Exception as e:  # noqa: BLE001 - the gold file is the record; MLflow is the index
        return f"not logged ({type(e).__name__}: {e})"


def _dataset(domain: str, exp_id: str) -> int:
    """`tau2-loop.<domain>.golden`: one record per conversation, gold-free inputs, golden expectations."""
    from mlflow.genai import datasets

    from tau2_loop.tooljudge import gold

    records = []
    for r in gold.read_gold(domain):
        a = r.get("answer") or {}
        records.append(
            {
                "inputs": {
                    "key": r["key"],
                    "task": r["task"],
                    "fold": r["fold"],
                    "half": r["half"],
                },
                "expectations": {
                    "passed": r["passed"],
                    "first_wrong_step": a.get("first_wrong_step"),
                    "verdicts": {c["id"]: c["verdict"] for c in a.get("checkpoints") or []},
                },
            }
        )
    name = dataset_name(domain)
    found = datasets.search_datasets(experiment_ids=exp_id, filter_string=f"name = '{name}'")
    ds = (
        found[0]
        if found
        else datasets.create_dataset(name=name, experiment_id=exp_id, tags={"project": PROJECT})
    )
    ds.merge_records(records)
    # a conversation that left the gold file (v0's, under the optimised-agents rule) leaves the set
    keep = {r["inputs"]["key"] for r in records}
    df = ds.to_df()
    stale = [
        str(rid)
        for rid, inputs in zip(df.get("dataset_record_id", []), df.get("inputs", []), strict=False)
        if isinstance(inputs, dict) and inputs.get("key") not in keep
    ]
    if stale:
        ds.delete_records(stale)
    return len(records)


def log_replay(run_dir: Path, summary: dict[str, Any]) -> str:
    """One J2 replay as an MLflow run; returns its id, or a note when MLflow refused."""
    import mlflow

    try:
        exp_id = _experiment()
        # a rescored replay updates its own run rather than adding a second one
        found = mlflow.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.mlflow.runName = '{run_dir.name}'",
            output_format="list",
        )
        resume = {"run_id": found[0].info.run_id} if found else {"run_name": run_dir.name}
        with mlflow.start_run(**resume) as run:
            mlflow.set_tags(
                {
                    **required_tags(),
                    "kind": "judge-replay",
                    "domain": str(summary.get("domain")),
                    "judge": str(summary.get("judge")),
                    "fingerprint": str(summary.get("fingerprint")),
                }
            )
            mlflow.log_params(
                {
                    "plan_judge_model": summary.get("model"),
                    "effort": summary.get("effort"),
                    "judge": summary.get("judge"),
                    "split": summary.get("split"),
                    "structured": summary.get("structured"),
                }
            )
            flat: dict[str, float] = {}
            for half, m in (summary.get("scores") or {}).items():
                for k, v in m.items():
                    if isinstance(v, int | float) and not isinstance(v, bool):
                        flat[f"{half}_{k}"] = float(v)
            for k, v in (summary.get("tokens") or {}).items():
                flat[f"tokens_{k}"] = float(v)
            mlflow.log_metrics(flat)
            for name in ("run.json", "summary.json", "verdicts.jsonl"):
                if (run_dir / name).is_file():
                    mlflow.log_artifact(str(run_dir / name))
            return str(run.info.run_id)
    except Exception as e:  # noqa: BLE001
        return f"not logged ({type(e).__name__}: {e})"


def log_cycle(entry: dict[str, Any]) -> str:
    """One J3 cycle as an MLflow run (`kind=judge-cycle`); returns its id, or a note when refused."""
    import json
    import tempfile

    import mlflow

    try:
        exp_id = _experiment()
        name = f"{entry['domain']}_cycle{entry['cycle']}_{entry['challenger']['name']}"
        gate = entry.get("gate") or {}
        with mlflow.start_run(run_name=name, experiment_id=exp_id) as run:
            mlflow.set_tags(
                {
                    **required_tags(),
                    "kind": "judge-cycle",
                    "domain": str(entry["domain"]),
                    "verdict": str(entry["verdict"]),
                    "champion": str(entry["champion"]["name"]),
                    "challenger": str(entry["challenger"]["name"]),
                }
            )
            mlflow.log_params(
                {
                    "cycle": entry["cycle"],
                    "champion_fingerprint": entry["champion"]["fingerprint"],
                    "challenger_fingerprint": entry["challenger"]["fingerprint"],
                    "optimiser_model": entry["optimiser"]["model"],
                    "optimiser_effort": entry["optimiser"]["effort"],
                    "gate_rule": gate.get("rule"),
                }
            )
            metrics: dict[str, float] = {
                "lessons_added": float(len(entry["lessons"]["added"])),
                "lessons_edited": float(len(entry["lessons"]["edited"])),
                "lessons_removed": float(len(entry["lessons"]["removed"])),
                "disagreements_read": float(entry["read"]["disagreements"]),
                "optimiser_input_tokens": float(entry["optimiser"]["input_tokens"]),
                "promoted": float(entry["verdict"] == "promoted"),
            }
            if gate:
                ba = gate["balanced_accuracy"]
                metrics.update(
                    {
                        "gate_fixed": float(len(gate["fixed"])),
                        "gate_broken": float(len(gate["broken"])),
                        "gate_p_value": float(gate["p_value"]),
                    }
                )
                for who in ("champion", "challenger"):
                    if ba[who] is not None:
                        metrics[f"gate_balanced_accuracy_{who}"] = float(ba[who])
                    if gate["read"][who] is not None:
                        metrics[f"read_balanced_accuracy_{who}"] = float(gate["read"][who])
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "cycle.json"
                p.write_text(json.dumps(entry, indent=1, ensure_ascii=False))
                mlflow.log_artifact(str(p))
            return str(run.info.run_id)
    except Exception as e:  # noqa: BLE001
        return f"not logged ({type(e).__name__}: {e})"
