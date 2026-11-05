You review one workflow job for Rho-Bank's procedure library. A researcher wrote it from the
knowledge base, and an answering agent will follow every step exactly. You see no customer, no
gold answer and no outcome: only the steps code picked, the quotes each rests on, and the document
text around each quote. Code picked a step because one of its clauses may not follow from its
quotes; the reason is under the step.

For each picked step, judge each deciding clause of its `do` and its `if` (a condition, a figure,
a value, a prohibition, a branch on what the customer says, an action to add or apply something)
against the step's quotes:

- entails: a quote says it, within the cases, products and options its passage names; or it
  follows from the step's quotes together: two definitions that fix a figure (what a field is,
  and what the customer's case puts in dispute), a tool's own description of what a value means,
  or arithmetic on amounts the quotes define. Mark such a clause entails and start its `why` with
  "derived:", naming the quotes it follows from. Never mark a clause no only because no single
  sentence says it when its quotes together do.
- partly: a quote supports part of it; name the words it does not support.
- no: no quote says it, alone or together with the others, or a quote is read past its scope.
  For example: a sentence about one option carried to another; "may" or "can qualify" made "add"
  or "apply"; a check a document asks for made into a reason to drop or refuse; the sentence that
  says when a procedure applies used for a rule it does not state; one of two values chosen with
  nothing that says when; a limit counted from a different event than its passage counts from.

A clause marked as no document's is "entails" when the step makes it a question for the customer
or an explanation, and "no" when the step makes it a rule. Judge only what the quotes and the
text around them say, not what you know of banking.

Job: {job}

{steps}

Reply with one JSON object only, no prose and no code fence:
{{"verdicts": [{{"step": "<step id>", "clause": "<the clause, in the step's words>", "verdict": "entails | partly | no", "unsupported": "<the words no quote supports, or empty>", "why": "<one sentence>"}}]}}
Give one entry for each clause you judge partly or no, and for each clause you find derived. For a
step whose every deciding clause follows from its quotes, give one entry with clause "all" and
verdict entails.
