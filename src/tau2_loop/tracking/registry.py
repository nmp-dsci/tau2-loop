"""The version registry, per domain: which agent folder is champion, which is challenger.

A JSON file committed with the repo (`loop/<domain>/registry.json`), mirrored
into the MLflow model registry as aliases when the server is up. The file is
the truth because the demo image has no MLflow and the gate runs in CI with no
server.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from tau2_loop.config import DOMAINS, registry_path, settings


def model_name(domain: str) -> str:
    return f"tau2-loop-{domain}"


def read_registry(domain: str) -> dict[str, Any]:
    p = registry_path(domain)
    if p.exists():
        return dict(json.loads(p.read_text()))
    return {"domain": domain, "champion": None, "challenger": None, "history": []}


def read_all() -> dict[str, dict[str, Any]]:
    return {d: read_registry(d) for d in DOMAINS}


def _write(domain: str, reg: dict[str, Any]) -> None:
    p = registry_path(domain)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2) + "\n")


def _entry(run_id: str) -> dict[str, Any]:
    from tau2_loop.eval.runner import load_run

    meta, _ = load_run(run_id)
    s = meta.summary or {}
    return {
        "agent": meta.agent,
        "fingerprint": meta.fingerprint,
        "run_id": run_id,
        "model": meta.model,
        "split": meta.split,
        "trials": meta.trials,
        "passed": s.get("passed"),
        "n_scored": s.get("n_scored"),
        "at": datetime.now(UTC).isoformat(),
    }


def register(run_id: str, alias: str = "challenger") -> dict[str, Any]:
    from tau2_loop.eval.runner import load_run

    meta, _ = load_run(run_id)
    reg = read_registry(meta.domain)
    e = _entry(run_id)
    reg[alias] = e
    reg.setdefault("history", []).append({"event": f"register:{alias}", **e})
    _write(meta.domain, reg)
    _mirror_alias(meta.domain, alias, e)
    return e


def promote(run_id: str) -> dict[str, Any]:
    from tau2_loop.eval.runner import load_run

    meta, _ = load_run(run_id)
    reg = read_registry(meta.domain)
    e = _entry(run_id)
    reg["champion"] = e
    if reg.get("challenger") and reg["challenger"].get("agent") == e["agent"]:
        reg["challenger"] = None
    reg.setdefault("history", []).append({"event": "promote", **e})
    _write(meta.domain, reg)
    _mirror_alias(meta.domain, "champion", e)
    return e


def champion_name(domain: str) -> str | None:
    c = read_registry(domain).get("champion")
    return str(c["agent"]) if c else None


def _mirror_alias(domain: str, alias: str, e: dict[str, Any]) -> None:
    """Best effort: the same alias on the MLflow model registry."""
    try:
        import mlflow
        from mlflow import MlflowClient

        from tau2_loop.eval.runner import load_run

        mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
        client = MlflowClient()
        name = model_name(domain)
        try:
            client.get_registered_model(name)
        except Exception:  # noqa: BLE001
            client.create_registered_model(name, description=f"tau2-loop {domain} agent versions")
        meta, _ = load_run(str(e["run_id"]))
        source = f"runs:/{meta.mlflow_run_id}/agent" if meta.mlflow_run_id else str(e["run_id"])
        mv = client.create_model_version(
            name,
            source=source,
            run_id=meta.mlflow_run_id,
            tags={"agent": str(e["agent"]), "fingerprint": str(e["fingerprint"])},
        )
        client.set_registered_model_alias(name, alias, mv.version)
    except Exception:  # noqa: BLE001 - the file is the record; the mirror is convenience
        return
