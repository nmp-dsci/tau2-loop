"""The published τ²-bench leaderboard, ingested into a committed index.

The board is not a service: it is a folder of `submission.json` files in the
upstream repo, one per entry, listed by `manifest.json`, which the taubench.com
site renders. An entry is self-reported — a team runs the harness itself and
opens a pull request — so the numbers here are exactly what each team claimed,
and `methodology.verification` is the only thing that distinguishes a standard
run from one with modified prompts.

We read that folder out of the pinned submodule and commit the result to
`data/index/leaderboard.json`, the same way DataAgentBench ingests its upstream:
the submodule is never edited, and the app must work from a bare clone without
it. Our own runs are joined to this table at serving time, never written into it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tau2_loop.config import DATA_DIR, DOMAINS, TAU2_ROOT

SUBMISSIONS_DIR = TAU2_ROOT / "web" / "leaderboard" / "public" / "submissions"
INDEX_DIR = DATA_DIR / "index"
LEADERBOARD_PATH = INDEX_DIR / "leaderboard.json"


def _entry(name: str, doc: dict[str, Any]) -> dict[str, Any]:
    """One submission, flattened to what a table row needs."""
    results = doc.get("results") or {}
    methodology = doc.get("methodology") or {}
    verification = methodology.get("verification") or {}
    scores: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        r = results.get(domain)
        if not isinstance(r, dict):
            continue
        scores[domain] = {
            # the site reports pass^k as a percentage; keep its own units
            **{f"pass_{k}": r.get(f"pass_{k}") for k in (1, 2, 3, 4)},
            "cost": r.get("cost"),
            "retrieval_config": r.get("retrieval_config"),
        }
    return {
        "id": name,
        "model": doc.get("model_name"),
        "model_org": doc.get("model_organization"),
        "submitted_by": doc.get("submitting_organization"),
        "date": doc.get("submission_date"),
        "type": doc.get("submission_type"),
        "modality": doc.get("modality"),
        "reasoning_effort": doc.get("reasoning_effort"),
        "user_simulator": methodology.get("user_simulator"),
        "tau2_version": methodology.get("tau2_bench_version"),
        "notes": methodology.get("notes"),
        "modified_prompts": verification.get("modified_prompts"),
        "omitted_questions": verification.get("omitted_questions"),
        "trajectories": bool(doc.get("trajectories_available")),
        "scores": scores,
        "domains": sorted(scores),
    }


def ingest(path: Path = LEADERBOARD_PATH) -> dict[str, Any]:
    """Read the vendored submissions folder and write the committed index."""
    manifest_path = SUBMISSIONS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no leaderboard manifest at {manifest_path} — `git submodule update --init` first"
        )
    listed = json.loads(manifest_path.read_text()).get("submissions") or []
    entries = []
    for name in listed:
        doc_path = SUBMISSIONS_DIR / name / "submission.json"
        if not doc_path.exists():  # listed but not committed upstream
            continue
        entries.append(_entry(name, json.loads(doc_path.read_text())))
    payload = {
        "source": "sierra-research/tau2-bench · web/leaderboard/public/submissions",
        "tau2_sha": _submodule_sha(),
        "listed": len(listed),
        "entries": entries,
    }
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def read(path: Path = LEADERBOARD_PATH) -> dict[str, Any]:
    if path.exists():
        return dict(json.loads(path.read_text()))
    return {"source": None, "tau2_sha": None, "listed": 0, "entries": []}


def best_per_domain(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The top `pass_1` per domain, for a page that wants one number to aim at."""
    best: dict[str, dict[str, Any]] = {}
    for e in entries:
        for domain, s in (e.get("scores") or {}).items():
            p1 = s.get("pass_1")
            if p1 is None:
                continue
            if domain not in best or p1 > best[domain]["pass_1"]:
                best[domain] = {"pass_1": p1, "id": e["id"], "model": e["model"]}
    return best


def _submodule_sha() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(TAU2_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
