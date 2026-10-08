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
  says so, in the edit's `why` or in its changelog;
- or a new job, as you drafted it, with `into` empty.

Write each job in one of two ways.

**As edits**, when it merges exactly one library job: name that job, and only it, in `into`, and
list what changes on its newest version above. The harness applies your edits to that version and
stores the result as its next version: whatever you do not edit is kept exactly as it is, so do not
copy it. The edits are the job's changelog; it needs no other.

  {{"job": "the name it keeps", "into": ["the library job"], "aliases": ["any new other name"],
   "edits": [
    {{"op": "add", "list": "steps", "after": "<step id>", "item": {{...the new step...}},
     "why": "the reason", "quote": "the document sentence, when one decides it", "doc": "doc id"}},
    {{"op": "change", "list": "rules", "key": "<the rule>", "fields": {{"quote": "...", "doc": "..."}},
     "why": "..."}},
    {{"op": "remove", "list": "tools", "key": "<tool name>", "why": "...", "quote": "...", "doc": "..."}},
    {{"op": "set", "field": "done_when", "value": "...", "why": "..."}}
   ]}}

- `list` is `steps` (named by `id`), `tools` (by `tool`), `info` (by `field`), `rules` (by `rule`:
  its text, or enough of its start to name one rule) or `aliases` (the name itself).
- `add` puts the `item` `after` or `before` the item named, or last; `change` gives the `fields` it
  sets or the whole new `item`; `remove` takes the item out; `set` replaces `when`, `starts`,
  `done_when` or another field outside those lists.
- A change or removal still needs a document saying the old version was wrong, in its `why` and
  `quote`. Edits name items as the version above names them; one that names nothing is sent back.

**In full**, when it merges two or more library jobs, or is new: the whole job in the shape your
instructions give, with `into` and `aliases` as above, and a `changelog`, the list of what this
merge did:

  "changelog": [{{"change": "added | changed | removed", "what": "the step id, tool name or rule",
                 "why": "the reason", "quote": "the document sentence, when one decides it",
                 "doc": "doc id"}}]

List every job you drafted or merged, in the order the answering agent should do them, plus
`documents` and `open_questions`. Code compares each job, the edits applied, with the newest version
(rubric D8: nothing lost that the edits or the changelog do not name) before taking it.
