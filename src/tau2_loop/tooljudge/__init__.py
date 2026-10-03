"""The tool judge (plan s11): an independent reviewer of the agent's plans, calibrated offline.

J0 `labels` — every checkpoint of every scored train conversation, labelled from gold, no model.
J1 `gold` — one golden answer per conversation, written by an annotator that sees gold.
J2 `prompt` · `core` · `replay` — the plan judge, replayed on every checkpoint, scored on J1.
"""
