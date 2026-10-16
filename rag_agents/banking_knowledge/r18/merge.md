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
  `into` listing every library job it merges and `aliases` every other name it goes by. A merge
  re-reads the whole version it merges, not only the steps this customer touches: earlier
  versions were written under older reading rules, and an old step is not right because it is
  there. For every step and rule of the version or versions it merges, read its quote and the
  document it cites, under the reading rules in your instructions, and do one of three:
  - **keep** it, when its quote and document say it as written, every deciding clause of it
    included (reading rule 19); name it in `kept`. A kept step faces the same checks as a new
    one: code picks the steps whose clauses may not follow from their quotes, and a reviewer
    reads them (D14). Many steps were written by r2 and r3, before reading rule 1 existed, and
    no merge since has read them against their passages: give a kept step its `basis` with a
    `change` edit when one quote stands behind several clauses;
  - **rewrite** it, when the documents say it differently: a condition turned into a bar, a
    reference point swapped for another, a parameter or case left out. Where your draft reads the
    same passage differently from the library, the reading that follows the reading rules
    replaces the other. Quote the passage the rewrite follows;
  - **remove** it, when no document supports it, and say in its `why` what you searched.
  Then add this customer's case: a new branch is a step with its own `if`. A step that serves
  another customer stays when a document supports it;
- or a new job, as you drafted it, with `into` empty.

Write each job in one of two ways.

**As edits**, when it merges exactly one library job: name that job, and only it, in `into`, and
list what changes on its newest version above. The harness applies your edits to that version and
stores the result as its next version: whatever you do not edit is kept exactly as it is, so do not
copy it, but name each step and rule you re-read and keep in `kept`. The edits are the job's
changelog; it needs no other.

  {{"job": "the name it keeps", "into": ["the library job"], "aliases": ["any new other name"],
   "kept": {{"steps": ["<step id>", "..."], "rules": ["<the rule, or enough of its start>", "..."]}},
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
- A rewrite quotes the passage it follows, in its `why` and `quote`; a removal says what you
  searched and found no support for. Edits name items as the version above names them; one that
  names nothing is sent back.

**In full**, when it merges two or more library jobs, or is new: the whole job in the shape your
instructions give, with `into` and `aliases` as above, `kept` as above for each step and rule
you re-read and copied unchanged, and a `changelog`, the list of what this merge did:

  "changelog": [{{"change": "added | changed | removed", "what": "the step id, tool name or rule",
                 "why": "the reason", "quote": "the document sentence, when one decides it",
                 "doc": "doc id"}}]

List every job you drafted or merged, in the order the answering agent should do them, plus
`documents`, `set_aside` (each document you printed in full that no job quotes, with why) and
`open_questions`. Code compares each job, the edits applied, with the newest version
(rubric D8: nothing lost that the edits or the changelog do not name; D9: every write step names
every parameter of its tool; D10: a write's documented requirements are met; D12: every step and
rule of the version it merges is kept and named in `kept`, rewritten, or removed; D13: a job that
calls or rules out a tool whose description points to the knowledge base cites a document that
names it) before taking it. It also sends back once, and records: a clause the reviewer finds
its quote does not say (D14), a document you printed in full that no job quotes and you did not
set aside (D15), and a document a called tool's description names by title that the job neither
cites nor sets aside (D16).
