"""Banking's local AllTools variants (s13, milestone 1), registered into tau2 from outside it.

tau2's AllTools retrieval is the leaderboard's: BM25, dense search and a read-only shell over
the knowledge base. Its dense search ships with two embedding backends, OpenAI and OpenRouter,
both paid web services the subscription cannot call. This module adds a third, `local`: a
sentence-transformers model on this machine, no key and no bill. It goes into the two lookup
tables tau2 builds embedding pipelines from, the document indexer's and the query encoder's,
which is what lets documents be embedded as they are and queries with the model's own query
prompt (Qwen3-Embedding has one; MiniLM has none). Then one variant per model, made with tau2's
own `all_tools_variant`, so each is AllTools exactly with only the embedder swapped.

A variant's name names its model, because a run records only the name (`RunMeta.retrieval`):
changing the model behind a name would make two runs that differ look alike. `register()` is
idempotent and runs before anything resolves a variant: the runner, replay and rescore call it.
tau2 is not modified; its tables are plain dicts it reads at call time.
"""

from __future__ import annotations

import threading
from typing import Any

from tau2_loop.config import ROOT

LOCAL_TYPE = "local"

# short key → Hugging Face model id. MiniLM is transcript-rag-agent's model; it reads only the
# first 256 tokens of what it embeds. Qwen3-Embedding-0.6B reads 32k.
EMBEDDING_MODELS: dict[str, str] = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}
# variant name → model key
LOCAL_VARIANTS: dict[str, str] = {
    "alltools_minilm": "minilm",
    "alltools_qwen3_0_6b": "qwen3-0.6b",
}
# tau2's default cache dir is relative to the working directory (`data/.embeddings_cache`),
# which would put vectors in this repo's committed data/; keep them in an ignored cache instead.
CACHE_DIR = ROOT / ".cache" / "embeddings"
BATCH_SIZE = 32

_lock = threading.Lock()
_models: dict[str, Any] = {}
_registered = False


def load_model(model_id: str) -> Any:
    """One SentenceTransformer per model per process, on the CPU: transcript-rag-agent found the
    Apple GPU path hangs on the first embed, and three conversations at once share one copy."""
    with _lock:
        m = _models.get(model_id)
        if m is None:
            from sentence_transformers import SentenceTransformer

            try:  # the copy already on disk, with no check against the Hub
                m = SentenceTransformer(model_id, device="cpu", local_files_only=True)
            except OSError:  # first use: download it once
                m = SentenceTransformer(model_id, device="cpu")
            _models[model_id] = m
        return m


def embed(model_id: str, texts: list[str], query: bool) -> Any:
    """Unit-length float32 vectors, one row per text; a query uses the model's `query` prompt
    when it has one."""
    import numpy as np

    m = load_model(model_id)
    prompt = "query" if query and "query" in (getattr(m, "prompts", None) or {}) else None
    with _lock:  # encode is not documented as thread-safe; queries are one short text each
        out = m.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name=prompt,
        )
    return np.asarray(out, dtype=np.float32)


def variant_model(name: str) -> str | None:
    """The Hugging Face model behind a local variant, else None."""
    key = LOCAL_VARIANTS.get(name)
    return EMBEDDING_MODELS[key] if key else None


def register() -> None:
    """Install the local embedder and variants into tau2; safe to call repeatedly."""
    global _registered
    with _lock:
        if _registered:
            return
        from tau2.domains.banking_knowledge import retrieval as tau2_retrieval
        from tau2.knowledge import embeddings_cache
        from tau2.knowledge.document_preprocessors import embedding_indexer
        from tau2.knowledge.embedders.base import BaseEmbedder
        from tau2.knowledge.input_preprocessors import embedding_encoder

        class LocalDocumentEmbedder(BaseEmbedder):  # type: ignore[misc]
            """tau2's embedder interface over a local model: documents, no prompt."""

            query = False

            def __init__(self, model: str, **_: Any) -> None:
                self.model = model

            def embed(self, texts: list[str]) -> Any:
                return embed(self.model, list(texts), query=self.query)

            def get_name(self) -> str:
                return f"{LOCAL_TYPE}:{self.model}"

        class LocalQueryEmbedder(LocalDocumentEmbedder):
            """The same model for a search query, with its query prompt."""

            query = True

        embedding_indexer.EMBEDDER_REGISTRY[LOCAL_TYPE] = LocalDocumentEmbedder
        embedding_encoder.EMBEDDER_REGISTRY[LOCAL_TYPE] = LocalQueryEmbedder
        if embeddings_cache._global_cache is None:  # noqa: SLF001 - tau2's own singleton
            embeddings_cache._global_cache = embeddings_cache.EmbeddingsCache(  # noqa: SLF001
                cache_dir=str(CACHE_DIR)
            )
        for name, key in LOCAL_VARIANTS.items():
            tau2_retrieval.RETRIEVAL_VARIANTS.setdefault(
                name,
                tau2_retrieval.all_tools_variant(
                    name, embedder_type=LOCAL_TYPE, embedder_model=EMBEDDING_MODELS[key]
                ),
            )
        _registered = True


def variant_tools(name: str | None) -> list[str]:
    """The knowledge tools a banking variant gives the agent, read from tau2's spec without
    building an environment (no sandbox, no embedding model); [] for an unknown name or None."""
    if not name:
        return []
    register()
    from tau2.domains.banking_knowledge import retrieval as tau2_retrieval

    v = tau2_retrieval.RETRIEVAL_VARIANTS.get(
        tau2_retrieval.RETRIEVAL_VARIANT_ALIASES.get(name, name)
    )
    if v is None:
        return []
    out: list[str] = []
    if v.kb_search is not None:
        out.append("KB_search")
    if v.kb_search_bm25 is not None:
        out.append("KB_search_bm25")
    if v.kb_search_dense is not None:
        out.append("KB_search_dense")
    if v.grep is not None:
        out.append("grep")
    if v.shell is not None:
        out.append("shell")
    return out


def variant_tool_rows(name: str | None) -> list[dict[str, Any]]:
    """The knowledge tools a variant gives, as the task extract lists a tool (`name`, the
    description tau2 builds from the tool's docstring, `type`, `mutates`), read from tau2's tool
    mixins without building an environment; [] for an unknown name or None."""
    import inspect

    from tau2.domains.banking_knowledge import retrieval_mixins
    from tau2.environment.tool import as_tool
    from tau2.environment.toolkit import MUTATES_STATE_ATTR, TOOL_TYPE_ATTR

    mixins = [c for _, c in inspect.getmembers(retrieval_mixins, inspect.isclass)]
    rows: list[dict[str, Any]] = []
    for tool in variant_tools(name):
        func = next((c.__dict__[tool] for c in mixins if tool in c.__dict__), None)
        if func is None:
            rows.append({"name": tool, "description": None, "type": "read", "mutates": False})
            continue
        kind = getattr(func, TOOL_TYPE_ATTR, None)
        rows.append(
            {
                "name": tool,
                "description": as_tool(func).openai_schema["function"]["description"],
                "type": getattr(kind, "value", "read"),
                "mutates": bool(getattr(func, MUTATES_STATE_ATTR, False)),
            }
        )
    return rows


def variant_summary(name: str | None) -> dict[str, Any]:
    """What the Agent tab shows for a version's retrieval: the variant, its tools, its dense
    model and the policy template it swaps in."""
    if not name:
        return {"variant": None, "tools": [], "dense_model": None, "template": None}
    register()
    from tau2.domains.banking_knowledge import retrieval as tau2_retrieval

    v = tau2_retrieval.RETRIEVAL_VARIANTS.get(
        tau2_retrieval.RETRIEVAL_VARIANT_ALIASES.get(name, name)
    )
    dense = None
    if v is not None:
        spec = v.kb_search_dense or (
            v.kb_search if v.kb_search is not None and v.kb_search.type == "embedding" else None
        )
        if spec is not None:
            dense = f"{spec.embedder_type}:{spec.embedder_model}"
    return {
        "variant": name,
        "tools": variant_tools(name),
        "dense_model": dense,
        "template": v.prompt_template.name if v is not None else None,
    }
