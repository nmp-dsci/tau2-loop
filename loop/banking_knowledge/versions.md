# banking_knowledge: every answering-agent version, in build order

The Optimise tab draws the 7 of 23 versions played on every task of their split: v1–v4, v7, v8, v23. The other 16 (v5, v6, v9–v22) were each played on part of a split, to check one change on the conversations it targeted, or never played, so a pass count of theirs cannot stand beside a full run's. They are how the optimisation went between the versions the tab draws, so they are kept here, between the versions they led to.

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
