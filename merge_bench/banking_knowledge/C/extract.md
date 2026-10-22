You break workflow jobs of Rho-Bank's procedure library into claims, so they can be checked one by
one against the documents and merged. Below are the library's version of the job "{job}" and a
new draft of it, written by a researcher from the knowledge base. Every step and rule has an
origin id: `L:` the library's version, `D:` the draft.

A claim is one thing the job tells the answering agent, small enough to be true or false on its
own:

- a value for a tool parameter in a case (`disputed_amount` for an incorrect amount is the
  difference between what was charged and what was expected);
- an action in a case (call a tool, transfer, stop, file one dispute per transaction);
- a condition or a limit (a time limit and the event it is counted from; a maximum; a status);
- a question to ask, and when;
- an order (what must happen before what).

Rules:

1. Every step and every rule of both versions gives at least one claim, and every deciding clause
   of its `do`, `if`, `args` and `basis` is in some claim. A step that says three things gives
   three claims. Two origins that say the same thing for the same case may share one claim: list
   both in `origin`.
2. `case` is the situation the claim applies to, in the shortest words that pick it out: `every
   dispute`, a dispute category such as `incorrect_amount`, `fraud with the card in hand`,
   `Rho-Bank ATM withdrawal`, `customer reports late`. Write the cases first, in `cases`, and give
   each claim one of them word for word, so claims about the same case meet.
3. `field` is what the claim decides: a tool parameter's name (`customer_max_liability_amount`),
   `action: <tool or what is done>`, `ask: <what>`, `order`, `condition`, `limit` or `message`.
   Two claims that decide the same thing for the same case get the same field, word for word, so
   a conflict between them shows.
4. `value` says what the claim says, in its own words, as a rule an agent could follow.
5. `quote` and `doc` are copied from the step or rule it comes from, the quote of the clause it
   rests on (a `basis` quote, or the step's `quote`); leave them empty when the clause has none,
   and copy a `"source": "none"` note into `note`.
6. Copy, do not judge: a claim you think is wrong is still a claim. Do not add claims neither
   version makes.

{versions}

Reply with one JSON object only, no prose and no code fence:
{{"cases": ["<case>", ...],
 "claims": [{{"id": "c1", "from": "library | draft", "origin": ["L:<id>", "D:<id>"], "case": "<one of cases>", "field": "<field>", "value": "<what it says>", "quote": "<copied, or empty>", "doc": "<doc id, or empty>", "note": "<a source none note, or empty>"}}]}}
