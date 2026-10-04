"""banking_knowledge's knowledge base, as the viewer names it: each document's title and size.

tau2 ships the knowledge base as one JSON file per document, `{id, title, content}`, under
`vendor/tau2-bench/data/tau2/domains/banking_knowledge/documents/` (698 of them). A task lists the
ones the agent needs as `required_documents`; Evals shows each by its title, with the full text
one click away. The demo image ships without tau2, so the titles and sizes are committed as a
small index (`data/tasks/banking_knowledge_documents.json`, `make documents`), and the text is
read from tau2 where it is present and is otherwise "not in this image".
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from tau2_loop.config import ROOT, TASKS_DIR, TAU2_DATA_DIR

KNOWLEDGE_DOMAINS = ("banking_knowledge",)


def source_dir(domain: str) -> Path:
    """Where tau2 keeps a domain's documents."""
    return TAU2_DATA_DIR / "tau2" / "domains" / domain / "documents"


def index_path(domain: str) -> Path:
    return TASKS_DIR / f"{domain}_documents.json"


def available(domain: str) -> bool:
    """Whether tau2's documents are on disk here (the demo image has none)."""
    return source_dir(domain).is_dir()


def build_index(domain: str = "banking_knowledge") -> dict[str, Any]:
    """Every document's title and character count, by id, read from tau2's files."""
    src = source_dir(domain)
    if not src.is_dir():
        raise FileNotFoundError(f"no documents at {src}: run `make setup` for the submodule")
    docs: dict[str, dict[str, Any]] = {}
    for p in sorted(src.glob("*.json")):
        d = json.loads(p.read_text())
        docs[str(d["id"])] = {"title": d.get("title"), "chars": len(d.get("content") or "")}
    return {
        "domain": domain,
        "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "n": len(docs),
        "documents": docs,
    }


def write_index(domain: str = "banking_knowledge") -> Path:
    p = index_path(domain)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_index(domain), indent=1, ensure_ascii=False) + "\n")
    return p


def read_index(domain: str) -> dict[str, dict[str, Any]]:
    """id → {title, chars}; empty for a domain with no knowledge base or no committed index."""
    p = index_path(domain)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        return {}
    return _index_at(str(p), mtime)


@lru_cache(maxsize=4)
def _index_at(path: str, mtime_ns: int) -> dict[str, dict[str, Any]]:
    return dict(json.loads(Path(path).read_text()).get("documents") or {})


def described(domain: str, ids: list[str]) -> list[dict[str, Any]]:
    """Each id with its title and size from the index, in the order given; unknown ids keep a
    null title rather than disappear."""
    index = read_index(domain)
    return [
        {
            "id": i,
            "title": (index.get(i) or {}).get("title"),
            "chars": (index.get(i) or {}).get("chars"),
        }
        for i in ids
    ]


def document(domain: str, doc_id: str) -> dict[str, Any] | None:
    """One document's full text from tau2's files: None when it is not there (an unknown id, or
    the demo image, which ships no tau2). Only a file directly in the documents folder is read."""
    src = source_dir(domain)
    p = src / f"{doc_id}.json"
    if "/" in doc_id or "\\" in doc_id or p.resolve().parent != src.resolve() or not p.is_file():
        return None
    d = json.loads(p.read_text())
    content = str(d.get("content") or "")
    return {
        "id": d.get("id", doc_id),
        "title": d.get("title"),
        "chars": len(content),
        "content": content,
    }
