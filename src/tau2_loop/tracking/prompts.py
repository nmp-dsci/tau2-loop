"""The agent's system prompt in MLflow's prompt registry, one version per fingerprint.

`agents/<domain>/<version>/system.md` is the record; this mirrors it so a run in
the tracking UI can be read back to the exact prompt text that produced it, and
so two fingerprints can be diffed there as well as here. PLATFORM.md lists the
prompt registry as one of the central server's four jobs.

The prompt template carries a `{policy}` slot that τ² fills with the domain's
policy document at run time; it is registered unfilled, because that slot is
what makes the prompt ours and the policy theirs.

Best effort throughout: the registry is an index, and a tracking server that is
down must never fail a promotion.
"""

from __future__ import annotations

from tau2_loop.config import settings


def prompt_name(domain: str) -> str:
    """`tau2-loop.airline.system` — the flat naming PLATFORM.md keeps for M1 projects."""
    return f"tau2-loop.{domain}.system"


def register_prompt(domain: str, version: str) -> str | None:
    """Register `agents/<domain>/<version>/system.md`; returns `name/version`, or None.

    Registering the same text twice is not an error: MLflow versions the prompt,
    and the `fingerprint` tag is what ties a version back to an agent folder.
    """
    try:
        import mlflow

        from tau2_loop.agent.versions import load_version

        v = load_version(domain, version)
        text = (v.files() or {}).get("system.md") or ""
        if not text.strip():
            return None
        mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
        registered = mlflow.genai.register_prompt(
            name=prompt_name(domain),
            template=text,
            commit_message=f"{domain}/{version} · fingerprint {v.fingerprint}",
            tags={
                "domain": domain,
                "agent": version,
                "fingerprint": v.fingerprint,
                "project": "tau2-loop",
            },
        )
        return f"{registered.name}/{registered.version}"
    except Exception:  # noqa: BLE001 - the agent folder is the record
        return None
