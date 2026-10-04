"""Each dataset's own optimiser (s13 §5): `optimisers/<domain>/`, read before every cycle.

Until s13 there was one optimiser: one prompt, one surface guide, one set of budgets, the model
and mode from the command line; only the data it read differed by dataset. A profile lets each
dataset's optimiser be tuned on its own, and every default is that one optimiser, so an untuned
dataset's prompts are byte for byte what they were (`tests/test_profiles.py`).

    optimisers/<domain>/profile.yaml   model, effort, mode, max_turns, the three failure-reading
                                       budgets, and the surfaces each mode may edit
    optimisers/<domain>/guide.md       the dataset's own lessons, appended to the shared prompt
                                       under their own heading when non-empty (every default:
                                       empty)
    optimisers/<domain>/guards.py      optional: `check(files, champion) -> list[str]`, extra
                                       checks on what the optimiser wrote, run beside the shared
                                       ones (`guards.profile_violations`), as read when the cycle
                                       began

A missing file, or a key missing from `profile.yaml`, is today's value. The harness reads the
profile and puts the guide in the prompt; the session is never pointed at `optimisers/`, and a
cycle that changes a file there is refused (`optimiser.GUARDED`). `--optimiser` and `--mode` on
`make loop` (`--optimiser` on `make ab`) still override a profile for one run, and the ledger
records the profile's fingerprint beside what was overridden (`OptimiserProfile.ledger_record`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import yaml

from tau2_loop.agent.versions import PROMPT_SURFACES, SURFACES
from tau2_loop.config import ROOT
from tau2_loop.llm import EFFORT, Effort, resolve_model

OPTIMISERS_DIR = ROOT / "optimisers"
# in name order: the order the fingerprint hashes them in
PROFILE_FILES = ("guards.py", "guide.md", "profile.yaml")
MODES = ("classic", "routing")

# Today's optimiser, which every profile defaults to.
MODEL = "opus"
MAX_TURNS = 120
# The failures section's budget, shared out per failure: a cycle with a handful (airline) keeps
# full lengths, one with sixty (banking's whole train) cuts each, and every failure names its
# trace so the session can read the whole conversation.
TRANSCRIPT_BUDGET = 60_000
SCENARIO_BUDGET = 60_000
DIFF_BUDGET = 90_000  # what tau2 graded: the database difference and the actions
# What each mode may edit: classic writes these; a routing diagnosis routes among these.
MODE_SURFACES: dict[str, tuple[str, ...]] = {"classic": PROMPT_SURFACES, "routing": SURFACES}
KEYS = (
    "model",
    "effort",
    "mode",
    "max_turns",
    "transcript_budget",
    "scenario_budget",
    "diff_budget",
    "surfaces",
)
_INTS = ("max_turns", "transcript_budget", "scenario_budget", "diff_budget")
EMPTY_FINGERPRINT = hashlib.sha256().hexdigest()[:12]  # a domain with no profile files


@dataclass(frozen=True)
class OptimiserProfile:
    """One dataset's optimiser. `OptimiserProfile(domain)` is today's optimiser."""

    domain: str
    model: str = MODEL
    effort: Effort = EFFORT
    mode: str = "classic"
    max_turns: int = MAX_TURNS
    transcript_budget: int = TRANSCRIPT_BUDGET
    scenario_budget: int = SCENARIO_BUDGET
    diff_budget: int = DIFF_BUDGET
    surfaces: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(MODE_SURFACES))
    guide: str = ""  # guide.md as written; the prompt takes it stripped, and only when non-empty
    guards: str | None = None  # optimisers/<domain>/guards.py's source, when there is one
    fingerprint: str = EMPTY_FINGERPRINT

    def surfaces_for(self, mode: str) -> tuple[str, ...]:
        return self.surfaces.get(mode, MODE_SURFACES[mode])

    def is_default(self) -> bool:
        """Today's optimiser: every setting its default, no guide and no guards of its own."""
        base = OptimiserProfile(self.domain)
        return (
            all(getattr(self, k) == getattr(base, k) for k in KEYS if k != "model")
            and resolve_model(self.model) == resolve_model(base.model)
            and not self.guide.strip()
            and self.guards is None
        )

    def ledger_record(self, model: str, mode: str) -> dict[str, Any]:
        """What a ledger entry keeps of the profile its cycle ran under (`optimiser_profile`):
        its fingerprint and settings, and in `overrides` the values the cycle ran with in their
        place (a flag on `make loop`, or the A/B pair's other mode)."""
        ran = {"model": model, "mode": mode}
        overrides = {
            k: v
            for k, v in ran.items()
            if (resolve_model(v) != resolve_model(self.model) if k == "model" else v != self.mode)
        }
        return {
            "domain": self.domain,
            "fingerprint": self.fingerprint,
            "tuned": not self.is_default(),
            "model": self.model,
            "effort": self.effort,
            "mode": self.mode,
            "max_turns": self.max_turns,
            "budgets": {
                "transcript": self.transcript_budget,
                "scenario": self.scenario_budget,
                "diff": self.diff_budget,
            },
            "surfaces": list(self.surfaces_for(mode)),
            "guide_chars": len(self.guide.strip()),
            "guards": self.guards is not None,
            "overrides": overrides,
        }


def profile_dir(domain: str) -> Path:
    return OPTIMISERS_DIR / domain


def fingerprint(domain: str) -> str:
    """sha256 of the profile files present, in name order, as `versions._fingerprint` hashes a
    version: a file added to the set later leaves every older profile's fingerprint as it was."""
    h = hashlib.sha256()
    d = profile_dir(domain)
    for name in PROFILE_FILES:
        p = d / name
        if p.is_file():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _surfaces(raw: Any, where: str) -> dict[str, tuple[str, ...]]:
    """`{classic: [...], routing: [...]}`, each a non-empty subset of what its mode can edit; a
    mode left out keeps today's. Stored in the surfaces' own order."""
    out = dict(MODE_SURFACES)
    if raw is None:
        return out
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: surfaces maps a mode ({', '.join(MODES)}) to its files")
    for mode, files in raw.items():
        if mode not in MODES:
            raise ValueError(f"{where}: surfaces names mode {mode!r}; the modes are {MODES}")
        if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
            raise ValueError(f"{where}: surfaces.{mode} is a non-empty list of file names")
        allowed = MODE_SURFACES[mode]
        bad = [f for f in files if f not in allowed]
        if bad:
            raise ValueError(
                f"{where}: {mode} may edit only {', '.join(allowed)}, not {', '.join(bad)}"
            )
        out[mode] = tuple(f for f in allowed if f in files)
    return out


def load_profile(domain: str) -> OptimiserProfile:
    """The domain's profile from `optimisers/<domain>/`: a missing file or key is today's value.
    A key it does not know, or a value out of range, is refused before any cycle starts."""
    d = profile_dir(domain)
    where = f"optimisers/{domain}/profile.yaml"
    p = d / "profile.yaml"
    raw = (yaml.safe_load(p.read_text()) if p.is_file() else None) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a mapping of {', '.join(KEYS)}")
    unknown = sorted(set(raw) - set(KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; a profile holds {', '.join(KEYS)}")
    base = OptimiserProfile(domain)
    model = raw.get("model", base.model)
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{where}: model is a model name (haiku, sonnet, opus or a full id)")
    effort = raw.get("effort", base.effort)
    if effort not in get_args(Effort):
        raise ValueError(f"{where}: effort is one of {get_args(Effort)}, not {effort!r}")
    mode = raw.get("mode", base.mode)
    if mode not in MODES:
        raise ValueError(f"{where}: mode is one of {MODES}, not {mode!r}")
    ints: dict[str, int] = {}
    for k in _INTS:
        v = raw.get(k, getattr(base, k))
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{where}: {k} is a positive whole number, not {v!r}")
        ints[k] = v
    guide = d / "guide.md"
    guards = d / "guards.py"
    return OptimiserProfile(
        domain=domain,
        model=model.strip(),
        effort=effort,
        mode=mode,
        surfaces=_surfaces(raw.get("surfaces"), where),
        guide=guide.read_text() if guide.is_file() else "",
        guards=guards.read_text() if guards.is_file() else None,
        fingerprint=fingerprint(domain),
        **ints,
    )
