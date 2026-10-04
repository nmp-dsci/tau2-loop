## Lessons from v3's diagnosis and the leaderboard (s14, 4 October 2026)

1. **Search depth is the leaders' edge.** The two leaders ran tau2's stock agent with this exact
   prompt: 8 to 13 knowledge look-ups before their first reply, about 24 a conversation, almost
   every needed tool name and most required documents read before their first write. The champion
   makes 2.6 and 10.7, replies without searching in a third of conversations, and has read half the
   required documents by its first write. In `system.md`, briefly: before verifying, acting or
   refusing, search the knowledge base for the customer's issue (BM25 and dense), open the best
   documents in full with the shell, and search again for every procedure step, tool name or
   document they name; act once each step of the plan has its document and its tool. A knowledge
   search is not customer data and needs no identity check.
2. **Search the step, not the topic.** The leaders found procedure tools with step queries
   ("check pending disputes … tool") and `grep -r <tool name>`; the champion searched the topic
   once and acted. Never tell a customer something cannot be done, or send them elsewhere, before
   searching for a tool that does it.
3. **A rule that can be checked goes in code, not the prompt.** `system.md` may grow 1,500
   characters: spend them on lessons 1, 2, 5 and 6. Put the rest in the code surfaces:
   - `memory.py`: `get_current_time`'s value; documents read in full and tool names (`name_1234`)
     seen in search and shell results, against those looked up, unlocked and called; the
     customer's requests for a human; which reads have returned (dispute history, pending
     replacement orders); accounts already credited.
   - `guidance.py`: before each model call, a short "leads not followed" note (tool names and
     documents named in results but never looked up), and from the fourth request for a human,
     the transfer rule.
   - `checks.py` `check_write`: `log_verification`'s `time_verified` must be `get_current_time`'s
     value; one combined credit per account; procedure order as a table of each write and the
     reads it needs (a credit card is not closed, logged or offered retention before its dispute
     history and pending replacements are read, nor closed with a replacement pending); no
     `call_discoverable_agent_tool` for a tool before a document naming it has been read.
   - `checks.py` `check_reply`: no refusal of a transfer once the customer has asked for a human
     four times (policy rule 6); a transfer needs no identity verification.
4. **Take v2's `helper.py`** (the held challenger's): `account_class` without the
   "(savings)"/"(checking)" suffix, a generic closure reason dropped so the tool records its
   default, `give_discoverable_user_tool` trimmed to the tool's name, the redacted-email guard.
   v2's `system.md` lifted train and lowered test: do not copy it wholesale.
5. **The champion makes native tool calls.** Its `system.md` still says "Always make sure you
   generate valid JSON only", an instruction for the JSON contract. Replace it: never write a tool
   call as text, and never say an action is done without its tool result.
6. **Judgement stays in the prompt, briefly.** Price every candidate product at the customer's
   stated usage from that product's own document; drop any that fails an eligibility rule; net
   several corrections into one; make only the writes the customer asked for; give a user tool
   with its exact argument names and values; invent no restriction the policy or the documents do
   not state.
7. **Cite the knowledge base, not the graded values.** The documents are in `.context/kb/`. A rule
   about a product, a fee or a dispute field must match its document (the dispute document's
   liability window runs from the statement, for example). Write nothing for a failure whose rule
   you cannot find in a document.
8. **Three "task faults" were not.** Failures blamed on the simulated customer were passed by a
   leader by searching before verifying, counting transfer requests, or making only the requested
   write. Mark a failure `none` only when no reading of the policy and documents would pass it.
