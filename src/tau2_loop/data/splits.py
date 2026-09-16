"""Our own train / test split per domain: `data/splits/<domain>.json`, cut once and committed.

tau2's `train` / `test` lists are fixed choices (and banking has none), so the
loop draws its own: `random.Random(300).sample(base_ids, 40)`, the first 20
train, the next 20 test, from the public `base` set of each scored domain. The
files are committed so every run, every cycle and every table reads the same
forty ids; `make splits` rewrites them only when the seed or the size changes.

Alongside each split the selected tasks themselves are extracted to
`data/tasks/<domain>.json` (the task spec as tau2 dumps it, plus the policy),
so the viewer and the demo image can show a task without the 140 MB submodule.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tau2_loop.config import DOMAINS, SPLIT_SEED, SPLIT_SIZE, SPLITS_DIR, TASKS_DIR, quiet_tau2

SPLITS = ("train", "test")


def _base_task_ids(domain: str) -> list[str]:
    quiet_tau2()
    from tau2.runner.helpers import load_task_splits, load_tasks

    splits = load_task_splits(domain)
    if splits and "base" in splits:
        return list(splits["base"])
    return [t.id for t in load_tasks(domain, None)]


def cut(domain: str, seed: int = SPLIT_SEED, size: int = SPLIT_SIZE) -> dict[str, Any]:
    ids = _base_task_ids(domain)
    if len(ids) < 2 * size:
        raise ValueError(f"{domain}: {len(ids)} base tasks, need {2 * size}")
    drawn = random.Random(seed).sample(ids, 2 * size)
    return {
        "domain": domain,
        "seed": seed,
        "size": size,
        "base_n": len(ids),
        "method": f"random.Random({seed}).sample(base_ids, {2 * size}); first {size} train, next {size} test",
        "train": drawn[:size],
        "test": drawn[size:],
        "reserve_n": len(ids) - 2 * size,
    }


def write_splits(seed: int = SPLIT_SEED, size: int = SPLIT_SIZE) -> list[Path]:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for domain in DOMAINS:
        split = cut(domain, seed, size)
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
    env_kwargs = {"retrieval_variant": "bm25"} if domain == "banking_knowledge" else {}
    env = build_environment(domain, env_kwargs=env_kwargs)
    tools = [
        {"name": t.name, "description": (t.openai_schema.get("function") or {}).get("description")}
        for t in env.get_tools()
    ]
    doc = {
        "domain": domain,
        "policy": env.get_policy(),
        "policy_words": len(env.get_policy().split()),
        "tools": tools,
        "tasks": [t.model_dump(mode="json") for t in tasks],
    }
    p = TASKS_DIR / f"{domain}.json"
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return p


def read_split(domain: str) -> dict[str, Any]:
    p = SPLITS_DIR / f"{domain}.json"
    if not p.exists():
        raise FileNotFoundError(f"no split for {domain}: run `make splits`")
    return dict(json.loads(p.read_text()))


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
