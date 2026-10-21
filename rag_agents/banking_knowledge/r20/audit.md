You audit one workflow job of Rho-Bank's procedure library against its documents. An answering
agent follows every step exactly. You see no customer, no draft, no gold answer and no outcome: only the job, the full text of every document it
cites, and the descriptions of the tools it names, which are the text the answering agent reads
when it unlocks a tool.

Judge each step and each rule. For each deciding clause of its `do`, its `if` and its `args` (a
condition, a figure, a value, a prohibition, a branch on what the customer says, an action to add
or apply something), decide which kind it is:

- stated: a passage says it, within the cases, products and options the passage names.
- derived: it follows from two or more passages, a tool's own description, or arithmetic on what
  they define. Write the derivation as short steps a checker can follow, each naming its passage.
  For example: a field is defined as "the dollar amount being disputed", and the case is defined
  as being charged a different amount than expected, so the amount in dispute is the difference.
- question: no document decides it, and the step asks the customer or explains instead of ruling.
- unsupported: none of these. No passage says it; a passage is read past its scope (a sentence
  about one option carried to another; "may" made "add" or "apply"; a check a document asks for
  made a reason to drop or refuse; the sentence that says when a procedure applies used for a rule
  inside it; one of two values chosen with nothing that says when; a limit counted from another
  event than its passage counts from); or a passage contradicts it.
- kept: a clause that forbids an action (an unlock, a reset, a closure, a transfer, any write) or
  holds back an offer, where no passage decides the case either way. A gap in the documents is not
  permission: the clause stays as written. A passage overrules it only when the passage names this
  case itself (its field value, code, product or condition), not when it reaches the case only
  through a category or heading the case falls under ("any automatic trigger", "scores above
  thresholds"); then the clause is unsupported and is rewritten to what that passage says.

Then give the step or rule its verdict:

- valid: every deciding clause is stated, derived, a question or kept.
- rewrite: one or more clauses are unsupported and the passages say what it should say instead.
  Write the step again, same id, in the job's shape: keep every clause that is stated, derived,
  a question or kept, replace each unsupported one with what the passages say, and give each
  deciding clause its `basis` entry (a quote, or `"source": "none"` and why, made a question).
  For a rule, write the rule again with its quote and basis.
- invalid: it rests on unsupported clauses and no passage gives a replacement; it is removed.

A step with one wrong clause beside right ones is rewritten, never removed: keep the right ones.
Two steps that give the same case opposite instructions are judged by the passage; the one it
does not support is rewritten or invalid. Judge only what the documents and the tool descriptions
say, not what you know of banking. Do not add new steps.

Job: {job}

The job's steps and rules (rules are numbered R1, R2, ...):

{version}

## The documents the job cites, in full

{documents}

## The tools the job names, as the answering agent reads them

{tools}

Reply with one JSON object only, no prose and no code fence:
{{"steps": [{{"step": "<step id>", "verdict": "valid | rewrite | invalid", "why": "<one sentence>",
  "clauses": [{{"clause": "<in the step's words>", "kind": "derived | kept | unsupported", "source": "<the derivation, or why it is unsupported>"}}],
  "rewrite": {{"id": "<same id>", "...": "the whole step, in the job's shape"}}}}],
 "rules": [{{"rule": "R<n>", "verdict": "valid | rewrite | invalid", "why": "<one sentence>",
  "rewrite": {{"rule": "...", "quote": "...", "doc": "...", "basis": []}}}}]}}
Give every step and every rule an entry. List in `clauses` only the clauses you judge derived,
kept or unsupported; for a kept clause, `source` names the passages that could be read either way.
Give `rewrite` only for verdict rewrite.
