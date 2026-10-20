# banking_knowledge: every answering-agent version, in build order

The Optimise tab draws the 7 of 29 versions played on every task of their split: v1–v4, v7, v8, v23. The other 22 (v5, v6, v9–v22, v24–v29) were each played on part of a split, to check one change on the conversations it targeted, or never played, so a pass count of theirs cannot stand beside a full run's. They are how the optimisation went between the versions the tab draws, so they are kept here, between the versions they led to.

Written by `make versions-log DOMAIN=banking_knowledge` from `agents/banking_knowledge/v*/`, `loop/banking_knowledge/ledger.jsonl`, `loop/banking_knowledge/registry.json` and the run folders, the reading `GET /api/versions` serves. Re-run it when a version is added or played; do not edit it by hand.

| version | on the tab | made | from | train | test | verdict |
|---|---|---|---|---|---|---|
| v1 | yes | model swap | v0 | 2/60 | 6/37 | first champion |
| v2 | yes | optimise | v1 | 17/60 | 5/37 | held (+3 −4) |
| v3 | yes | tool change | v1 | 12/60 | 6/37 | held (+2 −2) |
| v4 | yes | optimised | v3 | 29/60 | 14/37 | promoted (+8 −0) |
| v5 | no | optimise | v4 | 10 of 20 played | — | not gated |
| v6 | no | workflow tools | v4 | — | — | never played |
| v7 | yes | model swap | v6 | 37/60 | 12/37 | not gated |
| v8 | yes | workflow tools | v7 | 38/60 | 21/37 | promoted by hand |
| v9 | no | workflow tools | v8 | 1 of 2 played | 1 of 3 played | not gated |
| v10 | no | workflow tools | v8 | 1 of 5 played | 0 of 1 played | not gated |
| v11 | no | workflow tools | v8 | 1 of 5 played | 0 of 1 played | not gated |
| v12 | no | workflow tools | v8 | 1 of 5 played | 0 of 1 played | not gated |
| v13 | no | workflow tools | v8 | 1 of 1 played | 1 of 2 played | not gated |
| v14 | no | workflow tools | v8 | 2 of 2 played | 0 of 1 played | not gated |
| v15 | no | workflow tools | v8 | 1 of 2 played | 0 of 2 played | not gated |
| v16 | no | workflow tools | v8 | 0 of 3 played | 0 of 1 played | not gated |
| v17 | no | workflow tools | v8 | 2 of 3 played | 1 of 1 played | not gated |
| v18 | no | workflow tools | v8 | — | 1 of 3 played | not gated |
| v19 | no | workflow tools | v8 | — | 0 of 3 played | not gated |
| v20 | no | workflow tools | v8 | — | 1 of 3 played | not gated |
| v21 | no | workflow tools | v8 | — | 1 of 3 played | not gated |
| v22 | no | tool change | v21 | — | 2 of 3 played | not gated |
| v23 | yes | model swap | v22 | 43/60 | 20/37 | not gated |
| v24 | no | optimise | v23 | 5 of 17 played | 4 of 17 played | not gated |
| v25 | no | workflow tools | v23 | — | — | never played |
| v26 | no | workflow tools | v24 | 6 of 17 played | 5 of 17 played | not gated |
| v27 | no | tool change | v26 | 0 of 1 played | 3 of 17 played | not gated |
| v28 | no | workflow tools | v27 | 1 of 5 played | — | not gated |
| v29 | no | workflow tools | v27 | — | — | never played |

## v1 · on the Optimise tab

- **Made:** model swap from v0: model: haiku → sonnet
- **Played:** train 2 of 60 · test 6 of 37 · sonnet-5 · medium · first champion
- **Runs:** `20261002T235638Z_banking_knowledge_v1_train`, `20261003T003126Z_banking_knowledge_v1_test`

## v2 · on the Optimise tab

- **Made:** optimise cycle 1 from v1: opus optimiser
- **Played:** train 17 of 60 · test 5 of 37 · sonnet-5 · medium · held (+3 −4)
- **Why:** Added reply-format, identity/verification, and 11 working rules (KB use, exact names, eligibility before writes, op ordering, recommendation math, net credits, CC closure retention, debit dispute fields, identifiers, transfers, self-actions) around the unchanged policy slot.
- **Runs:** `20261003T005018Z_banking_knowledge_v2_train`, `20261003T051036Z_banking_knowledge_v2_test`

## v3 · on the Optimise tab

- **Made:** tool change cycle 2 from v1: retrieval: None → alltools_minilm; tool_mode: json → native; identity_note: False → True
- **Played:** train 12 of 60 · test 6 of 37 · sonnet-5 · medium · held (+2 −2)
- **Runs:** `20261004T023602Z_banking_knowledge_v3_train`, `20261004T023605Z_banking_knowledge_v3_test`

## v4 · on the Optimise tab

- **Made:** optimised cycle 3 from v3: effort: medium → high; parallel_calls: False → True
- **Played:** train 29 of 60 · test 14 of 37 · sonnet-5 · high · promoted (+8 −0)
- **Why:** Replaced the JSON-only line with native-call/no-false-claims rules and added search-first, date-from-get_current_time, whole-catalog recommendation, net-correction, request-ordering, do-what-was-asked, and debit-dispute-field lines (366 -> 1860 chars).
- **Runs:** `20261004T222002Z_banking_knowledge_v4_train`, `20261004T113341Z_banking_knowledge_v4_test`

## Between v4 and v7 · 2 versions left off the tab

### v5

- **Made:** optimise from v4: no ledger entry
- **Played:** train 10 of 20 played, of the split's 60 · no test run · sonnet-5 · high · not gated
- **Why:** Adds lesson 10's lines in its wording: the multistep line, the five-step write rubric, the 'only from a sentence you read' rule and the 'give a user tool first, full values' rule. Replaces 'disputed_amount = the full transaction' with doc _031's per-transaction fields (amount in error, type from the record, noticed date, possession and PIN per transaction) and keeps the liability sentence; 1860 to 3183 characters.
- **Runs:** `20261005T073300Z_banking_knowledge_v5_custom`

### v6

- **Made:** workflow tools from v4: workflows: None → r1
- **Played:** no train run · no test run · sonnet · high · never played
- **Why:** s16: the answering agent looks up workflow_rag's workflows (find_workflow) or asks it to research one (request_workflow); tells the model when to call the two workflow tools

## v7 · on the Optimise tab

- **Made:** model swap from v6: effort: high → medium
- **Played:** train 37 of 60 · test 12 of 37 · sonnet-5 · medium · not gated
- **Runs:** `20261006T163602Z_banking_knowledge_v7_train`, `20261006T163559Z_banking_knowledge_v7_test`

## v8 · on the Optimise tab · champion

- **Made:** workflow tools from v7: workflows: r1 → r3
- **Played:** train 38 of 60 · test 21 of 37 · sonnet-5 · medium · promoted by hand
- **Why:** Pointed at r3 before its first run (the person's call, 7 Oct 2026): r2's library plus the 26 questions r3 merged concurrently
- **Runs:** `20261008T134442Z_banking_knowledge_v8_train`, `20261008T134445Z_banking_knowledge_v8_test`

## Between v8 and v23 · 14 versions left off the tab

### v9

- **Made:** workflow tools from v8: workflows: r3 → r5
- **Played:** train 1 of 2 played, of the split's 60 · test 1 of 3 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D1: v8's prompt reading workflow_rag r5's library (research rule 1, rubric D10), to score D1's 5 failed conversations
- **Runs:** `20261009T071504Z_banking_knowledge_v9_custom`, `20261009T072105Z_banking_knowledge_v9_custom`

### v10

- **Made:** workflow tools from v8: workflows: r3 → r6
- **Played:** train 1 of 5 played, of the split's 60 · test 0 of 1 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D2: v8's prompt reading workflow_rag r6's library (rules 2-3, rubric D9), to score D2's 6 failed conversations
- **Runs:** `20261009T082751Z_banking_knowledge_v10_custom`, `20261009T084104Z_banking_knowledge_v10_custom`

### v11

- **Made:** workflow tools from v8: workflows: r3 → r7
- **Played:** train 1 of 5 played, of the split's 60 · test 0 of 1 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D2 again: v8's prompt reading workflow_rag r7's library (merges re-read every old step, rubric D12), to score D2's 6 failed conversations
- **Runs:** `20261009T100151Z_banking_knowledge_v11_custom`, `20261009T101359Z_banking_knowledge_v11_custom`

### v12

- **Made:** workflow tools from v8: workflows: r3 → r8
- **Played:** train 1 of 5 played, of the split's 60 · test 0 of 1 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D2 third step: v8's prompt reading workflow_rag r8's library (reading rules 4 and 5 on r7's re-reading merge), to score D2's 6 failed conversations
- **Runs:** `20261009T104037Z_banking_knowledge_v12_custom`, `20261009T105253Z_banking_knowledge_v12_custom`

### v13

- **Made:** workflow tools from v8: workflows: r3 → r9
- **Played:** train 1 of 1 played, of the split's 60 · test 1 of 2 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D3: v8's prompt reading workflow_rag r9's library (reading rules 6 and 7), to score D3's 3 failed conversations
- **Runs:** `20261009T111822Z_banking_knowledge_v13_custom`, `20261009T112414Z_banking_knowledge_v13_custom`

### v14

- **Made:** workflow tools from v8: workflows: r3 → r10
- **Played:** train 2 of 2 played, of the split's 60 · test 0 of 1 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D4: v8's prompt reading workflow_rag r10's library (reading rule 8 on r9), to score D4's 3 failed conversations
- **Runs:** `20261010T081042Z_banking_knowledge_v14_custom`, `20261010T081245Z_banking_knowledge_v14_custom`

### v15

- **Made:** workflow tools from v8: workflows: r3 → r11
- **Played:** train 1 of 2 played, of the split's 60 · test 0 of 2 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D5: v8's prompt reading workflow_rag r11's library (reading rule 9 on r10), to score D5's 4 failed conversations
- **Runs:** `20261010T083651Z_banking_knowledge_v15_custom`, `20261010T083958Z_banking_knowledge_v15_custom`

### v16

- **Made:** workflow tools from v8: workflows: r3 → r12
- **Played:** train 0 of 3 played, of the split's 60 · test 0 of 1 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D6: v8's prompt reading workflow_rag r12's library (reading rules 10 and 11 on r11), to score D6's 4 failed conversations
- **Runs:** `20261010T090045Z_banking_knowledge_v16_custom`, `20261010T090508Z_banking_knowledge_v16_custom`

### v17

- **Made:** workflow tools from v8: workflows: r3 → r13
- **Played:** train 2 of 3 played, of the split's 60 · test 1 of 1 played, of the split's 37 · sonnet-5 · medium · not gated
- **Why:** s22 D6 again: v8's prompt reading workflow_rag r13's library (tool descriptions shown to workflow_rag), to score D6's 4 failed conversations
- **Runs:** `20261010T094453Z_banking_knowledge_v17_custom`, `20261010T094818Z_banking_knowledge_v17_custom`

### v18

- **Made:** workflow tools from v8: workflows: r3 → r14
- **Played:** no train run · test 1 of 3 played, of the split's 37 · sonnet · medium · not gated
- **Why:** s22 D7: v8's prompt reading workflow_rag r14's library (reading rule 12 on r13), to score D7's 3 failed test conversations
- **Runs:** `20261011T003705Z_banking_knowledge_v18_custom`

### v19

- **Made:** workflow tools from v8: workflows: r3 → r15
- **Played:** no train run · test 0 of 3 played, of the split's 37 · sonnet · medium · not gated
- **Why:** s22 D7 again: v8's prompt reading workflow_rag r15's library (rule 10 follows a tool description's pointers), to score D7's 3 test conversations
- **Runs:** `20261011T012724Z_banking_knowledge_v19_custom`

### v20

- **Made:** workflow tools from v8: workflows: r3 → r16
- **Played:** no train run · test 1 of 3 played, of the split's 37 · sonnet · medium · not gated
- **Why:** s22 D7, option 3: v8's prompt reading workflow_rag r16's library (rubric D13: a tool's knowledge-base pointer followed), to score D7's 3 test conversations
- **Runs:** `20261011T024447Z_banking_knowledge_v20_custom`

### v21

- **Made:** workflow tools from v8: workflows: r3 → r17
- **Played:** no train run · test 1 of 3 played, of the split's 37 · sonnet · medium · not gated
- **Why:** s22 D7: v8 reading r17 (reading rule 13 and every named tool described), to score D7's test task_010, task_014 and task_015
- **Runs:** `20261011T041027Z_banking_knowledge_v21_custom`

### v22

- **Made:** tool change from v21: customer_tools: False → True
- **Played:** no train run · test 2 of 3 played, of the split's 37 · sonnet · medium · not gated
- **Why:** s22 D7: the answering agent never saw the tools the customer holds from the start; in test task_010 it tried to hand over submit_referral, then handed over get_referral_link. v22 = v21 plus the customer's own tools in its prompt, read from tau2
- **Runs:** `20261011T050525Z_banking_knowledge_v22_custom`

## v23 · on the Optimise tab

- **Made:** model swap from v22: model: sonnet → claude-sonnet-5-5
- **Played:** train 43 of 60 · test 20 of 37 · sonnet-5-5 · medium · not gated
- **Why:** the person's call: v22 with the answering agent on Sonnet 5.5 (claude-sonnet-5-5; v22's 'sonnet' is claude-sonnet-5), played with a Haiku 5.5 customer on v8's 20-task smoke before a full train and test run
- **Runs:** `20261011T055438Z_banking_knowledge_v23_train`, `20261011T055440Z_banking_knowledge_v23_test`

## After v23 · 6 versions left off the tab

### v24

- **Made:** optimise from v23: no ledger entry
- **Played:** train 5 of 17 played, of the split's 60 · test 4 of 17 played, of the split's 37 · sonnet-5-5 · medium · not gated
- **Why:** Debit-dispute liability is a tier only (50/500/-1), plus pin_compromised and signature vs PIN purchase; rewards-check method in the corrections bullet; recommendations drop products lacking a stated need or with a minimum above the held amount (+787 chars).
- **Runs:** `20261013T102726Z_banking_knowledge_v24_custom`, `20261013T141532Z_banking_knowledge_v24_custom`

### v25

- **Made:** workflow tools from v23: workflows: r17 → r18
- **Played:** no train run · no test run · sonnet-5-5 · medium · never played
- **Why:** s23 workflow arm: v23 reading r18 (r17 + W1-W7 and the per-clause provenance checks M1-M6; train 027 041 063 084 086 088 090 and R's 10 test questions researched)

### v26

- **Made:** workflow tools from v24: workflows: r17 → r18
- **Played:** train 6 of 17 played, of the split's 60 · test 5 of 17 played, of the split's 37 · sonnet-5-5 · medium · not gated
- **Why:** s23 combined arm (the person's call, 15 Oct 2026): v24's prompt and code reading r18's library; v25 skipped
- **Runs:** `20261015T041624Z_banking_knowledge_v26_custom`, `20261015T043217Z_banking_knowledge_v26_custom`

### v27

- **Made:** tool change from v26: workflows: r18 → r19; workflow_view: None → steps
- **Played:** train 0 of 1 played, of the split's 60 · test 3 of 17 played, of the split's 37 · sonnet-5-5 · medium · not gated
- **Why:** s25 bundle (the person's calls, 15 Oct 2026): v26 reading r19 (r18's drafts re-merged draft-first), find_workflow showing steps without quotes or basis (A2), the shown job kept in the hooks' state with a next-steps note (A1, A3), D2 accepting the amount under a tier (H1), a transfer before the 4th request sent back once (H2), transaction_type from doc 031's definitions (P1); 084: v24's line and hook D2 forced 500 over a $412.88 late charge; the tool's description weighs the disputed amount; 088: v24's 'pin_purchase only if the PIN may be compromised' read 'I don't know if the PIN was compromised' as pin_purchase; 084: in v24's run D2 sent the correct 412.88 back; no gold debit dispute in any run files 500 for an amount under $500; train replay: 43 transfers gold never makes, in 19 tasks, at a count under 4; policy rule 6; the person's ask for runtime workflow enforcement; test only (040, 085) shows the pattern, no train failure; A3 needs the calls already made
- **Runs:** `20261020T015825Z_banking_knowledge_v27_custom`, `20261015T170743Z_banking_knowledge_v27_custom`

### v28

- **Made:** workflow tools from v27: workflows: r19 → r19@20261020T000056Z_banking_knowledge_B_t1
- **Played:** train 1 of 5 played, of the split's 60 · no test run · sonnet-5-5 · medium · not gated
- **Why:** s30 E5 (the person's call, 20 Oct 2026): v27 reading r19's library with option B's trial-1 debit dispute job (merge bench) in place of r19 v3; held, scored on the job's train conversations only
- **Runs:** `20261020T015818Z_banking_knowledge_v28_custom`

### v29

- **Made:** workflow tools from v27: workflows: r19 → r19@20261020T213717Z_banking_knowledge_D-lean_t1
- **Played:** no train run · no test run · sonnet-5-5 · medium · never played
- **Why:** s32 E5 (the person's call, 21 Oct 2026): v27 reading r19's library with merge-bench option D-lean trial 1's debit dispute job in place of r19 v3; held, scored on the job's five train conversations only, as v28 was for option B
