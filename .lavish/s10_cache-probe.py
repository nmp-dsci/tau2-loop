"""s10 cache probe (run 2026-09-30, `uv run python .lavish/s10_cache-probe.py 8`): the same sealed options as core._query, but the raw usage printed.

Four Haiku calls: a tiny baseline, then a recorded banking prompt at agent call k,
the same prompt again, and the prompt at call k+2 (the transcript grown)."""
import asyncio, json, sys
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.domains.banking_knowledge.environment import get_environment
from tau2.utils.llm_utils import to_litellm_messages
from tau2_loop.agent.compose import compose
from tau2_loop.agent.versions import load_version
from tau2_loop.llm.core import check_billing, sealed_env, session_dir
from tau2_loop.llm.prompting import build_prompt

MODEL = "claude-haiku-4-5"

async def one(system, user):
    opts = ClaudeAgentOptions(model=MODEL, system_prompt=system, tools=[], allowed_tools=[],
        strict_mcp_config=True, permission_mode="bypassPermissions", max_turns=4,
        cwd=str(session_dir()), env=sealed_env(), setting_sources=[], effort="low")
    out = {}
    async for m in query(prompt=user, options=opts):
        if isinstance(m, ResultMessage):
            out = {"usage": m.usage, "cost": m.total_cost_usd, "turns": m.num_turns}
    return out

def main():
    check_billing()
    d = json.load(open("runs/20260929T042050Z_banking_knowledge_v1_train/tau2_results.json"))
    k0 = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sim = next(x for x in d["simulations"] if sum(1 for m in x["messages"] if m["role"] == "assistant" and m.get("usage")) >= k0 + 3)
    print("task", sim["task_id"])
    policy = d["info"]["environment_info"]["policy"]
    v = load_version("banking_knowledge", "v1")
    system_text = compose(v.system_prompt, policy, None).text
    env = get_environment(retrieval_variant="bm25_grep")
    tools = [t.openai_schema for t in env.get_tools()]
    cls = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    msgs = [cls[m["role"]].model_validate(m) for m in sim["messages"]]
    agent_idx = [i for i, m in enumerate(sim["messages"]) if m["role"] == "assistant" and m.get("usage")]
    def prompt_at(call):
        i = agent_idx[call]
        hist = [{"role": "system", "content": system_text}] + to_litellm_messages(msgs[:i])
        return build_prompt(hist, tools), sim["messages"][i]["usage"]["prompt_tokens"]
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    (sa, ua), rec_a = prompt_at(k)
    (sc, uc), rec_c = prompt_at(k + 2)
    print(json.dumps({"system_chars": len(sa), "user_chars_k": len(ua), "user_chars_k2": len(uc),
                      "recorded_sonnet_prompt_tokens_k": rec_a, "recorded_k2": rec_c,
                      "user_k2_starts_with_user_k": uc.startswith(ua[:-len('</conversation>')-1])}))
    runs = [("baseline: tiny system + 'hi'", "Reply with the word OK.", "hi"),
            ("A: banking call k", sa, ua), ("A again (identical)", sa, ua), ("C: call k+2 (grown)", sc, uc)]
    for label, s, u in runs:
        r = asyncio.run(one(s, u))
        print(label, json.dumps(r, default=str))

main()
