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

## Lessons from v4's diagnosis, the person's choices for v5 (s15, 5 October 2026)

9. **Start from v4 and keep it.** Keep v4's five surfaces, including checks A–G, `check_reply` and
   v2's helper. Change only what lessons 10–13 say. The model and effort are the harness's
   (Sonnet 5 at high); the person chose not to change them.
10. **A rubric at the write, in `system.md`.** In 24 of v4's 31 train failures the document or record
    that settled the step was already in the conversation, unused. Add these lines to v4's
    instructions, in this wording:
    - first, after the opening line: "This work is multistep: think carefully after every tool
      result, not only at the start."
    - after the search line: "Before every write or transfer, check in your reasoning: 1. Request:
      the customer asked for this, in their words (a KB "recommend" is advice, not a request).
      2. Procedure: you have read in full the document this step comes from, and looked up every tool
      and document it names. 3. Record: every condition it lists is checked against a tool result
      (accounts, transactions, disputes, pending orders), not the customer's word. 4. Arguments: each
      value comes from a tool result or the customer's words; none guessed, defaulted or partial.
      5. Order: doing it now blocks none of the customer's other requests. If any answer is no or
      unsure, search, read or ask first. One more search costs less than a wrong write."
    - "State a fee, rate, limit, eligibility or "can't" only from a sentence you read in a document;
      invent no restriction."
    - "Give a user tool (give_discoverable_user_tool) before telling the customer to use it, with
      every argument's full value."
    - Replace v4's "disputed_amount = the full transaction" with doc _031's amount in error: the full
      charge for fraud or unauthorized; the overcharge for an incorrect amount; the cash not
      dispensed for an ATM shortfall; one copy of a duplicate. The transaction type comes from the
      record, and the noticed or discovery date from when the customer first noticed. Keep v4's
      liability sentence (from the statement) as it is.
    No reflection send-back on first writes: the rubric lives in the prompt only.
11. **Targeted guardrails, each citing its document.** Each blocks once per key (v4's `_once`), so a
    wrong check costs one call and never traps a conversation.
    - `check_write` H, `close_debit_card` (doc _025 requirements 3–5, doc _026): blocked until
      `get_bank_account_transactions` has run for the card's account (any account when the card's is
      unknown); blocked when the customer asked to freeze and no card was frozen in this
      conversation; blocked while the card is frozen here and not unfrozen.
    - `check_write` B widened (logistics_003 Step 1): a `pay_credit_card_from_checking` while the
      customer has asked to close a credit card needs the dispute history and pending replacements
      read, as other closure-flow writes do. A payment for a limit increase is untouched.
    - `check_write` on transfers. Since s15, `check_write` also sees `transfer_to_human_agents`
      (`name == "transfer_to_human_agents"`, `arguments["reason"]`). Hold doc _042's reason codes and
      tiers in `checks.py` as data. When the customer raised an offer, promotion, flyer or letter
      the agent could not find and has asked for a human, a Tier 2–4 reason is sent back once
      naming `customer_demands_after_unavailable_offer_refusal`. Any other Tier 2–4 reason is sent
      back once with the Tier-1 list to re-check ("always select from the highest tier that
      applies"). After a tool result saying a record already exists (logistics_007), a transfer is
      sent back once: continue the procedure with the existing record (re-check eligibility, then
      decide).
    - New `check_write` rule checks: a credit-card dispute marked eligible for provisional credit
      beyond doc cc_015 item 4's count of disputes in the past 12 months (the history read plus those
      filed here) is sent back with the count. `update_transaction_rewards` is blocked unless a tool
      result shows that transaction RESOLVED or APPROVED (doc cc_004). `submit_referral` is blocked
      until the customer's accounts have been read after verification (doc _048).
    - `check_reply`: a reply naming a user tool (`name_1234`) not yet given, or a masked or partial
      id, together with an instruction to use it, is sent back once (give it first, full values). A
      reply with restriction wording (violat, misuse, against our policy, not permitted, not allowed,
      prohibited) is sent back once to cite the document that says it or drop it (policy rule 1).
      A referral-eligibility answer before the accounts are read is sent back once.
    - `guidance.py`, inside its 600 characters: after an "already exists" result, continue the
      procedure (do not transfer); once verified and before the accounts are read, read them before
      judging eligibility, tenure or balances; when the customer wants a recommendation, list the
      unread products of the category (from `ls` results and documents printed) before choosing;
      in the human-request note, doc _042's highest-tier rule.
    - `helper.py`: in `open_bank_account`, a class ending in "Saver" (case-insensitive) gets
      " Account" appended (doc _004 step 3), after the existing suffix strip.
    `memory.py` records what these need: transactions read by account, frozen and unfrozen cards,
    the customer's freeze, card-close and recommendation requests, tools given, transactions shown
    RESOLVED or APPROVED, the last "already exists" result, the category's products listed and read,
    accounts read, verified, and offers the customer raised.
12. **Not in v5.** No dispute-liability day window (the optimiser's 30-day D2), no forced provisional
    credit for required categories (D4), no amount-in-error block (D3): they fit the graded values of
    tasks no leader passes, and the knowledge base ties liability to the statement. No generic gate
    that blocks a write for unfollowed leads or a refusal without a search: replayed on v4's train
    conversations they fire as often on passes as on failures.
13. **Simulator faults stay `none`.** Failures where our simulated customer typed a tool call into
    its message, or never ran a tool it was given, are not the agent's: change nothing for them.
    Nor for a task whose gold omits an argument the tool requires.
