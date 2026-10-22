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

Write each drafted job again as one job made from the audited version and your draft; a drafted
job you named none for is a new job. You may search the documents again to settle a difference.
Merge only the jobs you named: the library holds only those, and a merge into another is sent
back. Then reply with one JSON object only, in the shape your instructions give, where each job
is:

- a merged job: the library job's name (or a better name from the document's title), with `into`
  listing every library job it merges and `aliases` every other name it goes by. Build it so:
  1. **The document decides, whoever wrote the step.** Neither side wins because it is newer or
     older. Where your draft and an audited step give the same case a different instruction,
     figure, condition or order, read the passage that decides that case and write what it says;
     when no passage decides it, the step asks the customer. Record each such case in the job's
     `conflicts` list: `{{"old": "<audited step id>", "draft": "<your step id>", "kept": "old |
     draft | both | question", "quote": "<the passage that decides it>", "doc": "<doc id>"}}`.
  2. **Every audited step stands unless something takes its place.** For each audited step your
     job no longer has (by id), add an entry to the job's `accounted` list:
     - `{{"step": "<old id>", "covered_by": "<the id of the step of your job that now handles its
       case>"}}` when a step of yours does what it did, for the same case; or
     - `{{"step": "<old id>", "replaced_by": {{"quote": "<the passage that decides its case
       otherwise>", "doc": "<doc id>"}}}}` when, under 1, a passage says something else.
     Code checks each: a cover must be a step of your job, and a replacing passage must be found in
     its document. An audited step you leave out unaccounted is sent back once, then carried into
     the job as it was.
  3. **Your draft's steps join where the procedure puts them.** Write the job in the order the
     answering agent should act, rewriting an audited step to take in what your draft adds for its
     case, never appending new logic after the end. One instruction per case (rule 11): no two
     steps give the same case different instructions. Your draft's steps face the same checks as
     every step (D14), and a deciding clause that follows from several quotes gives each in its
     `basis`.
  4. **What the audit removed stays out**, unless your research found a passage that supports it:
     then write it with that quote.

  Name in the changelog what you changed for a reason the documents give.
- or a new job, as you drafted it, with `into` empty.

Drop the `audit` marks: they are for you, not the answering agent. Write every job in full: the
whole job in the shape your instructions give, with `into`, `aliases`, `accounted` and `conflicts`
as above, and a `changelog`, the list of what this merge did with a reason:

  "changelog": [{{"change": "added | changed | removed", "what": "the step id, tool name or rule",
                 "why": "the reason", "quote": "the document sentence, when one decides it",
                 "doc": "doc id"}}]

List every job you drafted or merged, in the order the answering agent should do them, plus
`documents`, `set_aside` (each document you printed in full that no job quotes, with why) and
`open_questions`. Code checks each job before taking it (rubric D1: every quote is found in the
document it cites; D9: every write step names every parameter of its tool; D10: a write's
documented requirements are met; D13: a job that calls or rules out a tool whose description
points to the knowledge base cites a document that names it; D18: a step that names a period or
window says where it starts and ends; D19: every step id the job points to, in `tools`, `lookup`
or `after`, is a step of the job; D20: every write tool the job lists for the agent has a step that
calls it, or a `not_used` reason). It also sends back once, and records: a clause the reviewer
finds its quotes do not say (D14), an audited step left out unaccounted, and a document a called
tool's description names by title that the job neither cites nor sets aside (D16). After that one
send-back, an audited step still unaccounted is carried over, a quote still not found loses its
place in the job, a step the reviewer still finds unsupported is removed (except an audited valid
step you kept as the audit left it), and the job is taken; only D9, D10, D13 or D20 keep the
library's previous version.
