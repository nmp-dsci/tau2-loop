You grade one workflow job of Rho-Bank's procedure library against an answer key. An answering
agent serves customers by following this job exactly, as it is shown below (its steps, conditions,
arguments, rules and the facts it asks for; the quotes behind them are not shown to the agent).
For each fact in the key, decide whether the job leads the agent to it; for each trap, decide
whether the job tells the agent to do it. Judge only what the job says, not what a good agent
would know without it.

Facts:
- right: a step, an argument or a rule leads the agent to the fact for every case the fact names.
- partly: it does for some of the cases the fact names and not others, or it says it so loosely
  that an agent following it could give another answer.
- missing: nothing in the job leads the agent to it; the agent would have to work it out alone.
- wrong: the job leads the agent to a different answer for a case the fact names.

Traps:
- present: a step, an argument or a rule tells the agent to do it, for some case.
- absent: nothing in the job does.

Then list the conflicts: two steps, arguments or rules that give the same case different
instructions, so an agent could follow either.

Job: {job}

{workflow}

## The answer key

{facts}

## The traps

{traps}

Reply with one JSON object only, no prose and no code fence:
{{"facts": [{{"id": "K1", "verdict": "right | partly | missing | wrong", "where": ["<step id or rule>"], "evidence": "<at most 25 words of the job>", "why": "<one sentence>"}}],
 "traps": [{{"id": "T1", "present": true, "where": ["<step id or rule>"], "evidence": "<at most 25 words of the job>"}}],
 "conflicts": [{{"case": "<the case>", "where": ["<step id or rule>", "<step id or rule>"], "why": "<one sentence>"}}]}}
Give every fact and every trap an entry.
