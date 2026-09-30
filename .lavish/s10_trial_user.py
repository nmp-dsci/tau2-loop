"""s10 trial: the user simulator's cache split (Haiku 4.5, in-process on the same core).
  uv run python .lavish/s10_trial_user.py <trial run id>"""
import json
import sys
from pathlib import Path

d = json.load(open(Path(__file__).resolve().parents[1] / "runs" / sys.argv[1] / "tau2_results.json"))
raw = read = write = biggest = 0
for s in d["simulations"]:
    for m in s["messages"]:
        if m["role"] == "user" and m.get("usage"):
            u = (m.get("raw_data") or {}).get("usage") or {}
            raw += m["usage"]["prompt_tokens"]
            biggest = max(biggest, m["usage"]["prompt_tokens"])
            read += int(u.get("cache_read_input_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            write += int(u.get("cache_creation_input_tokens") or 0)
print(json.dumps({"raw": raw, "read": read, "write": write, "max": biggest}))
