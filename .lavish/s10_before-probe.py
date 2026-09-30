"""s10: the prompt as ONE text block (the path before fix A) on Sonnet 5, two consecutive calls,
to see what the old shape read back on the model the runs use. task_048, calls 21 and 22.
  uv run python .lavish/s10_before-probe.py
"""

import json

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.domains.banking_knowledge.environment import get_environment
from tau2.utils.llm_utils import to_litellm_messages

from tau2_loop.agent.compose import compose
from tau2_loop.agent.versions import load_version
from tau2_loop.llm.core import run_query
from tau2_loop.llm.prompting import build_prompt

d = json.load(open("runs/20260929T042050Z_banking_knowledge_v1_train/tau2_results.json"))
sim = next(s for s in d["simulations"] if s["task_id"] == "task_048")
system_text = compose(
    load_version("banking_knowledge", "v1").system_prompt, d["info"]["environment_info"]["policy"], None
).text
tools = [t.openai_schema for t in get_environment(retrieval_variant="bm25_grep").get_tools()]
cls = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
msgs = [cls[m["role"]].model_validate(m) for m in sim["messages"]]
at = [i for i, m in enumerate(sim["messages"]) if m["role"] == "assistant" and m.get("usage")]
for call in (20, 21):
    hist = [{"role": "system", "content": system_text}] + to_litellm_messages(msgs[: at[call]])
    system, user = build_prompt(hist, tools)  # one string: the old single-block shape
    r = run_query(system, user, "claude-sonnet-5", "medium")
    print(json.dumps({"shape": "one block", "call": call + 1, "input": r.input_tokens, "cache_read": r.cache_read,
                      "cache_write": r.cache_write, "output": r.output_tokens, "cost_usd": r.cost_usd}), flush=True)
