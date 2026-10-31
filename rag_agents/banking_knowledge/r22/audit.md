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
- question: no document decides it, and the step asks the customer for a fact or explains instead
  of ruling. Asking whether the customer still wants an action they asked for, or offering to drop
  it, is not a question: it is a condition that drops the action, judged under reading rule 6.
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

Six reading rules also decide a clause's kind. A clause that breaks one is unsupported, even when
a sentence read alone seems to state it, and is rewritten to what the rule and the passages give:

1. A field takes a value its document lists. When a document or a tool's description lists the
   values a field can take (tiers, codes, set amounts), the clause sets one of them by the
   document's own condition. A value computed from them (the lesser of two, a tier capped at the
   case's amount, the amount itself) is unsupported unless a passage states that arithmetic for
   that field; a description that says a value depends on something without saying how states
   none.
2. A condition comes from the step's own passage and the records. A condition that withholds,
   narrows or drops an action (only if, do not X when, must be Y to get Z, stop if the customer
   agrees) must be stated for that action by a passage the job cites, with its facts read from the
   records. One borrowed from another procedure or product, or one that turns on the customer's
   words or agreement where a record decides the fact, is unsupported: rewrite it to read the
   record, or to ask the customer for a fact no record holds.
3. The passage that names the case decides. When two passages give a case different instructions
   or values, the one that names the case (its field value, code, product, reason or condition)
   decides over one that reaches it only through a broader class or heading. A clause that follows
   the broader passage against the one that names the case is unsupported. This is the test `kept`
   applies to prohibitions, applied to every conflict. The two passages may sit in one document, as
   a list and its exceptions. An exception scoped to a class of the things the list names ("for
   non-fraud disputes" beside a list of dispute categories) does not override the list for a member
   the list names, unless the exception names that member too. An exception that applies to every
   case under a stated condition (a shared PIN, a new account) applies to the listed members too.
4. A rounding gap is not a discrepancy. A recorded figure equal to the amount times the right
   rate, rounded either way, is right. A clause that lists it as wrong, or offers to dispute or
   correct it, is unsupported, even where a document says how the system rounds: that passage says
   how a figure is computed, not that a figure rounded the other way is an error. A discrepancy is
   a recorded figure that implies another rate, category or promotion window.
5. A fee does not make a requirement optional. A balance a document says must be kept, or calls
   required (to stay open, in good standing, or to keep a rate or benefits), still holds when a
   document charges a fee for missing it. A clause that keeps a product whose requirement the
   customer will not meet, with the fee priced in, or that drops such a product only when no fee
   is charged, is unsupported: the product is dropped. A balance a document names only as what
   waives a fee is not a requirement.
6. No choice a document lacks. A step never asks the customer to opt in to part of their request,
   or offers to drop it, unless a passage gives the customer that choice for this case; a passage
   that gives it for one case (one category, one kind of record) does not give it for another. A
   clause that drops the action on the customer's answer without such a passage is unsupported,
   and is rewritten to act on what the records show. A customer who corrects a figure has not
   withdrawn the request.

A step that sets a field from a list of conditions (an eligibility, a category's action, a tier)
is judged case by case as well as clause by clause. For each value the documents list for the
field that decides it (each category, code or status the step's passages name), find the passage
that decides the outcome for that value, applying the rules above, and check that the step gives
that outcome. Each case whose outcome the step gets wrong is an unsupported clause: write the case
and the step's outcome as the `clause`, and the passage that decides it as the `source`.

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
