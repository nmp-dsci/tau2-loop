"""`make snapshot`: export the MLflow experiment to a JSON the demo image can serve.

The public deployment has no tracking server, so the runs page reads this
file. It is regenerated locally and committed with the run folders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tau2_loop.config import LOOP_DIR, settings

SNAPSHOT_PATH = LOOP_DIR / "mlflow_snapshot.json"


def write_snapshot(path: Path = SNAPSHOT_PATH) -> Path:
    import mlflow

    from tau2_loop.tracking.mlflow_log import EXPERIMENT

    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs: list[dict[str, Any]] = []
    if exp is not None:
        for r in mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list"):
            runs.append(
                {
                    "mlflow_run_id": r.info.run_id,
                    "name": r.info.run_name,
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "end_time": r.info.end_time,
                    "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
                    "params": dict(r.data.params),
                    "metrics": dict(r.data.metrics),
                }
            )
    payload = {
        "experiment": EXPERIMENT,
        "tracking_uri": settings().mlflow_tracking_uri,
        "runs": runs,
    }
    LOOP_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    return path


def read_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    if path.exists():
        return dict(json.loads(path.read_text()))
    return {"experiment": None, "runs": []}
