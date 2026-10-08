The library is holding the jobs you named. Merge: the library keeps one workflow per job, and each
new version must serve every customer the old one served and this customer too.

Your drafted jobs and the library jobs you named for them:

{pairs}

The library's newest versions of those jobs, in full. A version another researcher committed after
you read it above says so; merge into this one, not the one you read:

{current}

Merge each drafted job into the library jobs you named; a drafted job you named none for is a new
job. You may search the documents again to settle a difference. Merge only the jobs you named: the
library holds only those, and a merge into another is sent back. Then reply with one JSON object
only, in the shape your instructions give, where each job is:

- a merged job: the library job's name (or a better name from the document's title), with
  `into` listing every library job it merges and `aliases` every other name it goes by. It keeps
  every step, tool, rule and quote of the version or versions it merges, and adds this customer's
  case: a new branch is a step with its own `if`, not a change to the steps other customers need.
  It removes or changes something only when a document says the old version was wrong, and then
  says so in its changelog;
- or a new job, as you drafted it, with `into` empty.

Every job also carries a `changelog`, the list of what this merge did:

  "changelog": [{{"change": "added | changed | removed", "what": "the step id, tool name or rule",
                 "why": "the reason", "quote": "the document sentence, when one decides it",
                 "doc": "doc id"}}]

List every job you drafted or merged, in the order the answering agent should do them, plus
`documents` and `open_questions`. Write each merged job in full: the library stores what you write
as the job's next version, and code compares it with the newest one (rubric D8: nothing lost that
the changelog does not name) before taking it.
