The library is holding the jobs you named. Merge: the library keeps one workflow per job. Your
draft is the newest research on each job you drafted, so it is the base of the new version, and
the library's version is what it may take from.

Your drafted jobs and the library jobs you named for them:

{pairs}

The library's newest versions of those jobs, in full. A version another researcher committed after
you read it above says so; merge into this one, not the one you read:

{current}

Write each drafted job again, starting from your draft, merged with the library jobs you named; a
drafted job you named none for is a new job. You may search the documents again to settle a
difference. Merge only the jobs you named: the library holds only those, and a merge into another
is sent back. Then reply with one JSON object only, in the shape your instructions give, where
each job is:

- a merged job: the library job's name (or a better name from the document's title), with `into`
  listing every library job it merges and `aliases` every other name it goes by. Build it in this
  order:
  1. **Your draft first.** Every step, rule and branch of your draft stands. Where your draft and
     the library's version give a step, a figure, a condition or an order differently, your
     draft's wins. Earlier versions were written under older reading rules, and an old step is not
     right because it is there.
  2. **Then what the library's version has that your draft does not.** Carry an old step, rule or
     branch over when it covers a case no step of your draft covers (another customer's situation,
     another product, another branch), its quote is found in the document it cites, and every
     deciding clause follows from its quotes under the reading rules in your instructions (rule
     19). A clause follows from its quotes when one says it, or when they say it together: two
     definitions that fix a figure, a tool's own description of what a value means, or arithmetic
     on amounts they define. Give such a clause its `basis` with every quote it rests on. A carried
     step keeps its id and faces the same checks as your draft's (D14).
  3. **Account for every old step you leave out that stands on its documents.** An old step whose
     quotes are all found in the documents it cites stays unless you say why it goes. For each one
     your job no longer has (by id), add an entry to the job's `accounted` list:
     - `{{"step": "<old id>", "covered_by": "<the id of the step of your job that now handles its
       case>"}}` when a step of yours does what it did, for the same case; or
     - `{{"step": "<old id>", "replaced_by": {{"quote": "<the passage that decides its case
       otherwise>", "doc": "<doc id>"}}}}` when a passage says something else for its case.
     Code checks each: a cover must be a step of your job, and a replacing passage must be found in
     its document. An old step that stands on its documents and that you leave out unaccounted is
     sent back once, then carried into the job as it was. An old step whose quote is not found
     needs no account: code logs it.

  Name in the changelog what you changed for a reason the documents give.
- or a new job, as you drafted it, with `into` empty.

Write every job in full: the whole job in the shape your instructions give, with `into`,
`aliases` and `accounted` as above, and a `changelog`, the list of what this merge did with a
reason:

  "changelog": [{{"change": "added | changed | removed", "what": "the step id, tool name or rule",
                 "why": "the reason", "quote": "the document sentence, when one decides it",
                 "doc": "doc id"}}]

List every job you drafted or merged, in the order the answering agent should do them, plus
`documents`, `set_aside` (each document you printed in full that no job quotes, with why) and
`open_questions`. Code checks each job before taking it (rubric D1: every quote is found in the
document it cites, an old one carried over included; D9: every write step names every parameter
of its tool; D10: a write's documented requirements are met; D13: a job that calls or rules out a
tool whose description points to the knowledge base cites a document that names it; D18: a step
that names a period or window says where it starts and ends; D19: every step id the job points to,
in `tools`, `lookup` or `after`, is a step of the job; D20: every write tool the job lists for the
agent has a step that calls it, or a `not_used` reason). It also sends back once, and records: a
clause the reviewer finds its quotes do not say (D14), an old step left out unaccounted, and a
document a called tool's description names by title that the job neither cites nor sets aside
(D16). After that one send-back, an old step still unaccounted is carried over, a quote still not
found loses its place in the job, a step the reviewer still finds unsupported is removed, and the
job is taken; only D9, D10, D13 or D20 keep the library's previous version.
