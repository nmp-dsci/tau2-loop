"""s10 trial, step 2: fix A on five banking conversations against the same five before it.

Per agent call it reads the cache split the service now reports (raw_data.usage); the "before"
run's calls were all written fresh (the audit's probe), so every token there counts as a
1-hour write. Price-weighted uses the API ratios: plain input 1, 1-hour write 2, read 0.1.
  uv run python .lavish/s10_trial.py <trial run id> [<before run id>]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "20260929T042050Z_banking_knowledge_v1_train"
W1H, READ = 2.0, 0.1


def calls(sim):
    out = []
    for m in sim["messages"]:
        if m["role"] == "assistant" and m.get("usage"):
            u = ((m.get("raw_data") or {}).get("usage")) or {}
            det = u.get("prompt_tokens_details") or {}
            read = int(u.get("cache_read_input_tokens") or det.get("cached_tokens") or 0)
            write = int(u.get("cache_creation_input_tokens") or det.get("cache_creation_tokens") or 0)
            out.append(dict(inp=m["usage"]["prompt_tokens"], out=m["usage"]["completion_tokens"], read=read, write=write))
    return out


def conv(sim, measured):
    cs = calls(sim)
    raw = sum(c["inp"] for c in cs)
    if measured:
        read = sum(c["read"] for c in cs)
        write = sum(c["write"] for c in cs)
    else:  # before fix A every call wrote its whole prompt afresh
        read, write = 0, raw
    plain = raw - read - write
    return dict(task=sim["task_id"], reward=(sim.get("reward_info") or {}).get("reward") or 0.0,
                calls=len(cs), raw=raw, read=read, write=write, plain=plain,
                weighted=plain + write * W1H + read * READ, out=sum(c["out"] for c in cs),
                curve=[[c["inp"], c["read"], c["write"]] for c in cs] if measured else [[c["inp"]] for c in cs],
                dur=sim.get("duration"))


def main():
    trial = sys.argv[1]
    before = sys.argv[2] if len(sys.argv) > 2 else BEFORE
    t = json.load(open(ROOT / "runs" / trial / "tau2_results.json"))
    b = json.load(open(ROOT / "runs" / before / "tau2_results.json"))
    after = {s["task_id"]: conv(s, True) for s in t["simulations"]}
    prior = {s["task_id"]: conv(s, False) for s in b["simulations"] if s["task_id"] in after}
    print(json.dumps({"trial": trial, "before": before, "after": after, "prior": prior}, indent=1))


main()
