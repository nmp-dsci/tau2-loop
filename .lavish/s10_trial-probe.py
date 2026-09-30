"""s10 trial, step 1 (fix A): three sealed calls through core.run_query with the prompt as blocks.

The recorded banking conversation from the audit probe (task_002, banking v1 train), at agent
calls k, k+1 and k+2: each call's blocks start the next call's, so from the second call on the
cache should read what the call before wrote. Haiku 4.5 at effort low, as in the audit probe.
  uv run python .lavish/s10_trial-probe.py 8 [model] [effort] [task]
"""

import json
import sys

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.domains.banking_knowledge.environment import get_environment
from tau2.utils.llm_utils import to_litellm_messages

from tau2_loop.agent.compose import compose
from tau2_loop.agent.versions import load_version
from tau2_loop.llm.core import run_query
from tau2_loop.llm.prompting import build_blocks

RUN = "runs/20260929T042050Z_banking_knowledge_v1_train/tau2_results.json"


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    model = sys.argv[2] if len(sys.argv) > 2 else "claude-haiku-4-5"
    effort = sys.argv[3] if len(sys.argv) > 3 else "low"
    task = sys.argv[4] if len(sys.argv) > 4 else "task_002"
    d = json.load(open(RUN))
    sim = next(s for s in d["simulations"] if s["task_id"] == task)
    system_text = compose(
        load_version("banking_knowledge", "v1").system_prompt, d["info"]["environment_info"]["policy"], None
    ).text
    tools = [t.openai_schema for t in get_environment(retrieval_variant="bm25_grep").get_tools()]
    cls = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    msgs = [cls[m["role"]].model_validate(m) for m in sim["messages"]]
    at = [i for i, m in enumerate(sim["messages"]) if m["role"] == "assistant" and m.get("usage")]
    for call in (k, k + 1, k + 2):
        hist = [{"role": "system", "content": system_text}] + to_litellm_messages(msgs[: at[call]])
        system, blocks = build_blocks(hist, tools)
        r = run_query(system, blocks, model, effort)
        print(json.dumps({"task": task, "model": model, "effort": effort, "call": call + 1, "blocks": len(blocks), "chars": sum(map(len, blocks)),
                          "input": r.input_tokens, "cache_read": r.cache_read, "cache_write": r.cache_write,
                          "output": r.output_tokens, "cost_usd": r.cost_usd, "error": r.error}), flush=True)


main()
