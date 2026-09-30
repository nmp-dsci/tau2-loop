"""Our own train / test split per domain: `data/splits/<domain>.json`, cut once and committed.

tau2's `train` / `test` lists are fixed choices (and banking has none), so the
loop draws its own from the public `base` set of each scored domain. Version 1
drew `random.Random(300).sample(base_ids, 40)`: the first 20 train, the next 20
test, the rest in reserve. Version 2 (`SPLIT_VERSION`) halves the whole base
set and keeps what was held out, held out: every version-1 train task stays in
train, every version-1 test task stays in test, and the reserve, shuffled with
the same seed, is dealt out evenly — an odd one goes to test. The file keeps
the version-1 lists under `v1`, so a run on the old cut can be told apart. The
files are committed so every run, every cycle and every table reads the same
ids; `make splits` rewrites them only when the seed or the version changes.

Alongside each split the selected tasks themselves are extracted to
`data/tasks/<domain>.json` (the task spec as tau2 dumps it, plus the policy),
so the viewer and the demo image can show a task without the 140 MB submodule.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tau2_loop.config import (
    BANKING_RETRIEVAL,
    DOMAINS,
    HALVED_DOMAINS,
    SPLIT_SEED,
    SPLIT_VERSION,
    SPLITS_DIR,
    TASKS_DIR,
    TEST_CAP,
    V1_SIZE,
    quiet_tau2,
)

SPLITS = ("train", "test")


def _base_task_ids(domain: str) -> list[str]:
    quiet_tau2()
    from tau2.runner.helpers import load_task_splits, load_tasks

    splits = load_task_splits(domain)
    if splits and "base" in splits:
        return list(splits["base"])
    return [t.id for t in load_tasks(domain, None)]


def cut_ids(ids: list[str], seed: int = SPLIT_SEED, v1_size: int = V1_SIZE) -> dict[str, Any]:
    """The version-2 cut of a base id list: version 1's draw kept, the reserve dealt out evenly."""
    if len(ids) < 2 * v1_size:
        raise ValueError(f"{len(ids)} base tasks, need {2 * v1_size}")
    drawn = random.Random(seed).sample(ids, 2 * v1_size)
    v1 = {"train": drawn[:v1_size], "test": drawn[v1_size:]}
    held = set(drawn)
    reserve = [t for t in ids if t not in held]
    dealt = random.Random(seed).sample(reserve, len(reserve))
    half = len(dealt) // 2  # an odd reserve puts its extra task on the reported side
    return {
        "train": v1["train"] + dealt[:half],
        "test": v1["test"] + dealt[half:],
        "v1": v1,
    }


def cut(domain: str, seed: int = SPLIT_SEED) -> dict[str, Any]:
    ids = _base_task_ids(domain)
    c = cut_ids(ids, seed)
    split: dict[str, Any] = {
        "domain": domain,
        "version": SPLIT_VERSION,
        "seed": seed,
        "base_n": len(ids),
        "method": (
            f"v1: random.Random({seed}).sample(base_ids, {2 * V1_SIZE}), first {V1_SIZE} train, "
            f"next {V1_SIZE} test; v2 keeps both and deals the remaining {len(ids) - 2 * V1_SIZE} "
            f"out with random.Random({seed}).sample(reserve, n), first half train, the rest test"
        ),
        "train": c["train"],
        "test": c["test"],
        "reserve_n": len(ids) - len(c["train"]) - len(c["test"]),
        "v1": c["v1"],
        **({"halves": cut_halves(c["train"], seed)} if domain in HALVED_DOMAINS else {}),
    }
    return cap_test(split, TEST_CAP[domain]) if domain in TEST_CAP else split


def cap_test(split: dict[str, Any], n: int) -> dict[str, Any]:
    """Test cut to its first `n` tasks — v1's test first, then the dealt ones — and the rest held
    back in reserve. Nothing moves to train: a task that was ever test is never trained on."""
    full = list(split["test"]) + list((split.get("test_cap") or {}).get("held_back") or [])
    kept, held = full[:n], full[n:]
    out = dict(split)
    out["test"] = kept
    out["reserve_n"] = int(split.get("base_n") or 0) - len(split["train"]) - len(kept)
    out["test_cap"] = {
        "n": n,
        "method": f"the first {n} of the v2 test list (v1's {len(split['v1']['test'])} test tasks, then the dealt ones); the other {len(held)} held back",
        "held_back": held,
    }
    return out


def cut_halves(train: list[str], seed: int = SPLIT_SEED) -> dict[str, Any]:
    """Train dealt into a read half, whose failures the optimiser reads, and a gate half the gate
    decides on and no optimiser is shown (s09 option B). Train and test keep their members."""
    dealt = random.Random(seed).sample(train, len(train))
    half = len(dealt) // 2
    return {
        "seed": seed,
        "method": f"random.Random({seed}).sample(train, {len(train)}): first {half} read, the rest gate",
        "read": dealt[:half],
        "gate": dealt[half:],
    }


def add_halves(domain: str, seed: int = SPLIT_SEED) -> Path:
    """Write a domain's read and gate halves into its committed split, leaving every other key as it was."""
    p = SPLITS_DIR / f"{domain}.json"
    split = read_split(domain)
    split["halves"] = cut_halves(list(split["train"]), seed)
    p.write_text(json.dumps(split, indent=2) + "\n")
    return p


def halves(domain: str) -> tuple[list[str], list[str]] | None:
    """(read, gate): the halves of a domain's train split, or None where train is not halved."""
    if domain == "mock":
        return None
    h = read_split(domain).get("halves")
    return (list(h["read"]), list(h["gate"])) if h else None


def write_splits(seed: int = SPLIT_SEED) -> list[Path]:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for domain in DOMAINS:
        split = cut(domain, seed)
        p = SPLITS_DIR / f"{domain}.json"
        p.write_text(json.dumps(split, indent=2) + "\n")
        out.append(p)
        _write_task_extract(domain, split["train"] + split["test"])
    return out


def _write_task_extract(domain: str, ids: list[str]) -> Path:
    from tau2.runner.build import build_environment
    from tau2.runner.helpers import load_tasks

    wanted = set(ids)
    tasks = [t for t in load_tasks(domain, None) if t.id in wanted]
    # the variant the runs use (s09: bm25_grep), so the extract's tools and policy are the agent's
    env_kwargs = {"retrieval_variant": BANKING_RETRIEVAL} if domain == "banking_knowledge" else {}
    env = build_environment(domain, env_kwargs=env_kwargs)
    doc = {
        "domain": domain,
        "policy": env.get_policy(),
        "policy_words": len(env.get_policy().split()),
        "tools": [_tool_row(env.tools, t) for t in env.get_tools()],
        # the customer's own tools (telecom's phone, banking's app): the user simulator calls these
        "user_tools": [_tool_row(env.user_tools, t) for t in env.get_user_tools()]
        if env.user_tools is not None
        else [],
        "tasks": [t.model_dump(mode="json") for t in tasks],
    }
    p = TASKS_DIR / f"{domain}.json"
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return p


def _tool_row(kit: Any, tool: Any) -> dict[str, Any]:
    """A tool as the viewer shows it: `type` is tau2's own read / write / think / generic,
    `mutates` whether replaying it changes the database (what a state rebuild re-runs)."""
    return {
        "name": tool.name,
        "description": (tool.openai_schema.get("function") or {}).get("description"),
        "type": kit.tool_type(tool.name).value,
        "mutates": bool(kit.tool_mutates_state(tool.name)),
    }


def read_split(domain: str) -> dict[str, Any]:
    p = SPLITS_DIR / f"{domain}.json"
    if not p.exists():
        raise FileNotFoundError(f"no split for {domain}: run `make splits`")
    return dict(json.loads(p.read_text()))


def split_version(domain: str) -> int | None:
    """The committed cut's version; None for `mock`, which has no split file."""
    if domain == "mock":
        return None
    return int(read_split(domain).get("version", 1))


def split_ids(domain: str, split: str) -> list[str]:
    """The task ids of `train` or `test` for a domain; `all` is both; `mock` has no split file."""
    if domain == "mock":
        from tau2.runner.helpers import load_tasks

        return [t.id for t in load_tasks("mock", None)]
    if split == "all":
        s = read_split(domain)
        return list(s["train"]) + list(s["test"])
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS + ('all',)}, got {split!r}")
    return list(read_split(domain)[split])


def read_task_extract(domain: str) -> dict[str, Any]:
    p = TASKS_DIR / f"{domain}.json"
    if not p.exists():
        return {"domain": domain, "policy": "", "policy_words": 0, "tools": [], "tasks": []}
    return dict(json.loads(p.read_text()))


def tool_kinds(domain: str) -> dict[str, str]:
    """Each domain tool's tau2 type (`read`, `write`, `generic`), from the committed extract."""
    try:
        return {t["name"]: str(t.get("type")) for t in read_task_extract(domain).get("tools", [])}
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return {}


def apply_test_cap(domain: str) -> Path:
    """Cap a domain's committed test split in place (`TEST_CAP`), every other key as it was."""
    p = SPLITS_DIR / f"{domain}.json"
    p.write_text(json.dumps(cap_test(read_split(domain), TEST_CAP[domain]), indent=2) + "\n")
    return p
