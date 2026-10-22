The library is holding the jobs you named. Merge: the library keeps one workflow per job. Before
you read it, the library's version was audited against its documents by a reviewer in its own
context: each step below is marked `"audit": "valid"` (every deciding clause is stated by a
passage, derived from passages, or a question for the customer) or `"audit": "rewritten"`
(rewritten to what its passages say). The steps and rules the audit found unsupported are removed
and listed after the versions, with why.

Your drafted jobs and the library jobs you named for them:

{pairs}

The library's newest versions of those jobs, as the audit left them. A version another researcher
committed after you read it above says so; merge into this one, not the one you read:

{current}

Do not search the documents again: your research is done, and the auditor read every document the
version cites in full. Your tools are closed for this reply; a call is refused. Write only the
jobs listed above, each drafted job made one with the audited version it names: the library holds
only those, and any other job of your research is left out of this reply (code drops it unread).
Reply with one JSON object only, in the shape your instructions give, where each job is:

- a merged job, written as **edits on the audited version above**: the library job's name (or a
  better name from the document's title) in `job`, that library job, and only it, in `into`, and
  `aliases` for every other name it goes by. Code applies your edits to the audited version (its
  `audit` marks dropped) and checks the job that comes out: whatever you do not edit is kept
  exactly as it is, so do not copy it. Decide the edits so:
  1. **The document decides, whoever wrote the step.** Neither side wins because it is newer or
     older. Where your draft and an audited step give the same case a different instruction,
     figure, condition or order, read the passage that decides that case and write what it says;
     when no passage decides it, the step asks the customer. Record each such case in the job's
     `conflicts` list: `{{"old": "<audited step id>", "draft": "<your step id>", "kept": "old |
     draft | both | question", "quote": "<the passage that decides it>", "doc": "<doc id>"}}`.
  2. **Every audited step stands unless something takes its place.** A `remove` of an audited step
     says what does: `"covered_by": "<the id of the step of your job that now handles its case>"`
     when a step of yours does what it did, for the same case; or the `quote` and `doc` of the
     passage that, under 1, decides its case otherwise. Code checks each: a cover must be a step of
     the job, and a passage must be found in its document. An audited step removed without either
     is sent back once, then carried into the job as it was.
  3. **Your draft's steps join where the procedure puts them.** Change an audited step to take in
     what your draft adds for its case, or add a step `after` or `before` the step it belongs
     beside, in the order the answering agent should act; never append new logic after the end.
     One instruction per case (rule 11): no two steps give the same case different instructions.
     Your draft's steps face the same checks as every step (D14), and a deciding clause that
     follows from several quotes gives each in its `basis`.
  4. **What the audit removed stays out**, unless your research found a passage that supports it:
     then add it with that quote.

  The edits are the job's changelog; it needs no other. Each edit carries its `why`, and the
  `quote` and `doc` of the passage when one decides it:

  {{"job": "the name it keeps", "into": ["the library job"], "aliases": ["any new other name"],
   "conflicts": [ ...as in 1... ],
   "edits": [
    {{"op": "add", "list": "steps", "after": "<step id>", "item": {{ ...the new step, in the shape
     your instructions give... }}, "why": "...", "quote": "...", "doc": "..."}},
    {{"op": "change", "list": "steps", "key": "<step id>", "fields": {{"if": "...", "args": {{...}}}},
     "why": "...", "quote": "...", "doc": "..."}},
    {{"op": "remove", "list": "steps", "key": "<step id>", "covered_by": "<step id>", "why": "..."}},
    {{"op": "remove", "list": "rules", "key": "<the rule>", "why": "...", "quote": "...", "doc": "..."}},
    {{"op": "set", "field": "done_when", "value": "...", "why": "..."}}
   ]}}

  - `list` is `steps` (named by `id`), `tools` (by `tool`), `info` (by `field`), `rules` (by
    `rule`: its text, or enough of its start to name one rule) or `aliases` (the name itself).
  - `add` puts the `item` `after` or `before` the item named, or last; `change` gives the `fields`
    it sets (a field you give replaces that field whole) or the whole new `item`; `remove` takes
    the item out; `set` replaces `when`, `starts`, `done_when` or another field outside those
    lists.
  - Edits name items as the audited version above names them, or as an earlier edit of yours
    added them. An edit that names nothing is sent back; a job whose edits still do not apply
    keeps the library's previous version.
- or a new job (a drafted job you named no library job for), written in full in the shape your
  instructions give, with `into` empty and a `changelog`.

List the jobs above, in the order the answering agent should do them, plus `documents`,
`set_aside` (each document you printed in full that no job quotes, with why) and
`open_questions`. Code checks each job before taking it (rubric D1: every quote is found in the
document it cites; D9: every write step names every parameter of its tool; D10: a write's
documented requirements are met; D13: a job that calls or rules out a tool whose description
points to the knowledge base cites a document that names it; D18: a step that names a period or
window says where it starts and ends; D19: every step id the job points to, in `tools`, `lookup`
or `after`, is a step of the job; D20: every write tool the job lists for the agent has a step that
calls it, or a `not_used` reason). It also sends back once what fails, and records: a clause the
reviewer finds its quotes do not say (D14), an audited step removed without what takes its place,
and a document a called tool's description names by title that the job neither cites nor sets
aside (D16). After that one send-back, an audited step still unaccounted is carried over, a quote
still not found loses its place in the job, and the auditor reads the merged job against its whole
documents: a clause they do not support is rewritten to what they say, a step none of them supports
is removed unless the job needs it to pass these checks, and the job is taken; only D9, D10, D13
or D20 keep the library's previous version.
