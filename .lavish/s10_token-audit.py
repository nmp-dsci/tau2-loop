"""s10 token audit: where an agent conversation's input tokens come from, airline vs banking,
and what each lever would have saved on the same recorded calls.

Reads runs/<id>/tau2_results.json only; no model call. Prints the JSON the s10 page is drawn from:
  uv run python .lavish/s10_token-audit.py 20260928T060029Z_airline_v3_train \
      20260929T042050Z_banking_knowledge_v1_train 20260915T121036Z_banking_knowledge_v0_train
"""
import json, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo root: this file sits in .lavish/
KB = ("KB_search", "grep")
DOC = re.compile(r"(?m)^\d+\. .*\n   ID: (\S+)\n")
STUB = 90  # chars left for a doc that is dropped: its number, title and ID
# API list ratios, input = 1: a 1-hour cache write, a cache read, an output token (Sonnet 5 in:out 1:5 assumed)
W1H, READ, OUT = 2.0, 0.1, 5.0
# the system prompt + tools a call reads back from the cache even in the one-block shape, measured
# on Sonnet 5 (.lavish/s10_before-probe.py); 0 where it was not measured
SYS_READ = {"banking_knowledge": 5226}

def chars(m):
    c = len(m.get("content") or "")
    if m.get("tool_calls"):
        c += len(json.dumps([{"id": t["id"], "name": t["name"], "arguments": t["arguments"]} for t in m["tool_calls"]]))
    return c + 30  # role tag and blank lines around each turn

def docs(text):
    """(doc_id, chunk chars) for each document a KB_search/grep result lists."""
    ms = list(DOC.finditer(text))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.append((m.group(1), end - m.start()))
    return out

def conv(s):
    msgs = s["messages"]; names = {}
    items = []  # dicts: src, chars, at (agent-call index when added), docs
    calls = []  # per agent call: dict of numbers
    n_call = 0
    for m in msgs:
        if m["role"] == "assistant" and m.get("usage"):
            u, o = m["usage"]["prompt_tokens"], m["usage"]["completion_tokens"]
            calls.append(dict(u=u, o=o, items=[dict(x) for x in items], k=n_call))
            n_call += 1
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["name"]
            src = "assistant"
        elif m["role"] == "user":
            src = "user"
        elif m["role"] == "tool":
            nm = names.get(m.get("id"), "user tool")
            src = "kb" if nm in KB else "tool"
        else:
            src = m["role"]
        it = dict(src=src, chars=chars(m), at=n_call)
        if src == "kb":
            it["docs"] = docs(m.get("content") or "")
        items.append(it)
    return calls

def scenario_chars(items, how, k):
    """Transcript chars a call would carry under a lever."""
    seen = set(); total = 0
    for it in items:
        c = it["chars"]
        if it["src"] == "kb" and how != "now":
            ds = it.get("docs") or []
            if how == "dedupe":
                for d, n in ds:
                    if d in seen: c -= n - STUB
                    seen.add(d)
            elif how == "elide":  # a result older than the last two agent calls keeps titles and IDs only
                if k - it["at"] >= 2:
                    c -= sum(n - STUB for _, n in ds)
            elif how == "top5":
                c -= sum(n for _, n in ds[5:])
            elif how == "dedupe+elide":
                for d, n in ds:
                    if d in seen or k - it["at"] >= 2: c -= n - STUB
                    seen.add(d)
        total += c
    return total

def run(run_id):
    d = json.load(open(ROOT / "runs" / run_id / "tau2_results.json"))
    sys_read = SYS_READ.get(d["info"]["environment_info"]["domain_name"], 0)
    convs = []
    for s in d["simulations"]:
        calls = conv(s)
        if not calls: continue
        first = calls[0]["u"]
        comp = defaultdict(float); lev = defaultdict(float)
        prev_u = prev_d = None
        for c in calls:
            u, o, k = c["u"], c["o"], c["k"]
            sr = min(sys_read, u)
            items = c["items"]
            base = scenario_chars(items, "now", k) or 1
            by = defaultdict(int)
            for it in items: by[it["src"]] += it["chars"]
            comp["system"] += first
            for src, n in by.items(): comp[src] += (u - first) * n / base
            dd = 0.0
            for how in ("dedupe", "elide", "top5", "dedupe+elide"):
                v = first + (u - first) * scenario_chars(items, how, k) / base
                lev[how] += v
                if how == "dedupe": dd = v
            # dedupe shortens a result when it arrives, so the earlier prompt stays a prefix: caching still reads it
            lev["w_cached_dedupe"] += (sr * READ + (dd - sr) * W1H) if prev_d is None else (prev_d * READ + max(dd - prev_d, 0) * W1H)
            prev_d = dd
            lev["now"] += u
            lev["out"] += o
            # price-weighted: now the system prompt is read and the rest written afresh at 1h;
            # cache-friendly reads the last call's whole prompt and writes the new turns
            lev["w_now"] += sr * READ + (u - sr) * W1H
            lev["w_nocache"] += u * 1.0
            lev["w_cached"] += (sr * READ + (u - sr) * W1H) if prev_u is None else (prev_u * READ + max(u - prev_u, 0) * W1H)
            prev_u = u
        convs.append(dict(task=s["task_id"], reward=(s.get("reward_info") or {}).get("reward") or 0,
            calls=len(calls), a_in=sum(c["u"] for c in calls), a_out=sum(c["o"] for c in calls),
            first=first, last=calls[-1]["u"], curve=[c["u"] for c in calls],
            comp=dict(comp), lev=dict(lev),
            kb_calls=sum(1 for it in calls[-1]["items"] if it["src"] == "kb"),
            kb_docs=sum(len(it.get("docs") or []) for it in calls[-1]["items"] if it["src"] == "kb"),
            kb_unique=len({dd for it in calls[-1]["items"] if it["src"] == "kb" for dd, _ in it.get("docs") or []}),
            u_in=sum(m["usage"]["prompt_tokens"] for m in s["messages"] if m["role"] == "user" and m.get("usage")),
            u_out=sum(m["usage"]["completion_tokens"] for m in s["messages"] if m["role"] == "user" and m.get("usage")),
            dur=s.get("duration")))
    return d, convs

def summary(run_id):
    d, cs = run(run_id)
    n = len(cs); mean = lambda k: sum(c[k] for c in cs) / n
    comp = defaultdict(float); lev = defaultdict(float)
    for c in cs:
        for k, v in c["comp"].items(): comp[k] += v / n
        for k, v in c["lev"].items(): lev[k] += v / n
    L = max(len(c["curve"]) for c in cs)
    curve = []
    for i in range(L):
        vals = [c["curve"][i] for c in cs if len(c["curve"]) > i]
        if len(vals) >= 5: curve.append([i + 1, int(st.median(vals)), len(vals)])
    return dict(run=run_id, n=n, passed=sum(1 for c in cs if c["reward"] >= 1),
        model=d["info"]["agent_info"]["llm"], calls=mean("calls"), a_in=mean("a_in"), a_out=mean("a_out"),
        u_in=mean("u_in"), u_out=mean("u_out"), first=mean("first"), last=mean("last"),
        per_call=sum(c["a_in"] for c in cs) / sum(c["calls"] for c in cs),
        kb_calls=mean("kb_calls"), kb_docs=mean("kb_docs"), kb_unique=mean("kb_unique"),
        dur_median=st.median(c["dur"] for c in cs if c["dur"]),
        pass_calls=st.mean([c["calls"] for c in cs if c["reward"] >= 1] or [0]),
        fail_calls=st.mean([c["calls"] for c in cs if c["reward"] < 1] or [0]),
        pass_in=st.mean([c["a_in"] for c in cs if c["reward"] >= 1] or [0]),
        fail_in=st.mean([c["a_in"] for c in cs if c["reward"] < 1] or [0]),
        comp=dict(comp), lev=dict(lev), curve=curve,
        points=[[c["calls"], c["a_in"], 1 if c["reward"] >= 1 else 0, c["task"]] for c in cs])

if __name__ == "__main__":
    print(json.dumps([summary(r) for r in sys.argv[1:]], indent=1))
