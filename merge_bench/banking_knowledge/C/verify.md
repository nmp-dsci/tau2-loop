You verify claims about one job of Rho-Bank's procedure library, "{job}", against the documents.
Two versions of the job, the library's and a researcher's new draft, were broken into claims and
grouped by case and field. An answering agent will follow the job written from the claims you
keep. You see no customer, no gold answer and no outcome: only the claims, the full text of every
document they cite, and the descriptions of the tools they name, which are the text the answering
agent reads when it unlocks a tool. Which version a claim came from does not matter: a claim is
kept because the documents back it, never because it is newer or older.

For each claim, decide its kind:

- stated: a passage says it, within the cases, products and options the passage names.
- derived: it follows from two or more passages, a tool's own description, or arithmetic on what
  they define. Write the derivation as short steps a checker can follow, each naming its passage.
  For example: a field is defined as "the dollar amount being disputed", and the case is defined
  as being charged a different amount than expected, so the amount in dispute is the difference.
- question: no document decides it, and the claim asks the customer or explains instead of
  ruling; or it rules on something no document decides and can be kept as a question for the
  customer (say so in `why`).
- unsupported: none of these. No passage says it; a passage is read past its scope (a sentence
  about one option carried to another; "may" made "add" or "apply"; a check a document asks for
  made a reason to drop or refuse; the sentence that says when a procedure applies used for a rule
  inside it; one of two values chosen with nothing that says when; a limit counted from another
  event than its passage counts from); or a passage contradicts it.

A claim whose quote is marked NOT FOUND is judged on the documents alone. Then settle every slot
marked as having two or more values: read the passage that decides that case and keep the claim
or claims it supports (two values can both stand when they apply to different sub-cases; say
which); when no passage decides between them, make the slot a question for the customer. Judge
only what the documents and tool descriptions say, not what you know of banking.

## The claims, by case and field

{slots}

## The documents the claims cite, in full

{documents}

## The tools the claims name, as the answering agent reads them

{tools}

Reply with one JSON object only, no prose and no code fence:
{{"claims": [{{"id": "c1", "kind": "stated | derived | question | unsupported", "derivation": "<for derived: the steps, each with its passage; else empty>", "quote": "<the passage that decides it, copied, when it is not the claim's own>", "doc": "<its doc id>", "why": "<one sentence>"}}],
 "slots": [{{"slot": "<case> · <field>, as headed above", "keep": ["c1"], "question": false, "quote": "<the passage that decides it>", "doc": "<doc id>", "why": "<one sentence>"}}]}}
Give every claim an entry, and every marked slot an entry.
