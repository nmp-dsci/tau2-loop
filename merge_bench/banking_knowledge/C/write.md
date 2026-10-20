You write one workflow job of Rho-Bank's procedure library, "{job}", from verified claims. Two
versions of the job, the library's and a researcher's new draft, were broken into claims, and each
claim was checked against the documents. The claims below are the ones that survived: each is
stated by a passage, derived from passages (its derivation is given), or a question for the
customer. An answering agent will follow every step you write exactly.

Write the job from scratch, in the order the answering agent should act:

1. Every surviving claim is in the job, carried by the step or rule it decides. List the ids of
   the claims a step carries in its `claims` field (`"claims": ["c4", "c17"]`); a rule likewise.
   No step may carry a claim that is not below.
2. A claim of kind question is written as a question to the customer or an explanation, with a
   basis entry `"source": "none"`, never as a rule.
3. A derived claim keeps its derivation: give each passage it follows from as a `basis` quote on
   the clause, so a reviewer who reads the step's quotes can follow it.
4. Build the steps the procedure needs from the claims: verification and lookups first, then the
   checks, then the questions each case needs, then each write with every parameter of its tool
   (rule 3: name where each value comes from, by case where the value depends on the case), then
   what follows the write. A case's value for a parameter goes in the write step's `args`, by
   case. Merge claims that make one step into that step; never append a claim as a step of its
   own after the end. One instruction per case (rule 11).
5. Keep `when`, `starts`, `info`, `tools` and `done_when` consistent with the claims, taken from
   the two versions' own (given below): every tool a step calls is in `tools` with that step's
   id, and every tool listed for the agent that no step calls says why in `not_used`.
6. Every quote is copied character for character from the claim it comes with (or from the
   passages of a derivation); do not add quotes the claims do not have.

Write the job in the shape and under the reading rules below.

## The surviving claims

Each line: id [where it came from] · kind: what it says · quote (doc) · derivation.

{claims}

## How the slots with two values were settled

{settled}

## The two versions' names, `when`, `starts`, `info`, `tools` and `done_when`

{meta}

## The tools, as the answering agent reads them

{tools}

## The reading rules and the job's shape

{schema}

Reply with one JSON object only, no prose and no code fence: {{"jobs": [<the one job, with
`claims` on each step and rule>]}}.
