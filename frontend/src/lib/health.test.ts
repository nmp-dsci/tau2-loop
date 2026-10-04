/**
 * A run's harness health on Runs (s13), pinned: every number carries its denominator, the
 * knowledge-tool rows appear only for a run that has knowledge tools, and a dataset without
 * required documents shows a dash, not a number. The fixture is banking v1's train run as
 * `GET /api/runs/<id>` returns its `health`.
 */
import { describe, expect, it } from 'vitest';
import type { RunHealth } from './api';
import { healthClaim, healthRows, usesKnowledge } from './health';

const V1_TRAIN: RunHealth = {
  conversations: 60,
  passed: 2,
  replies: 641,
  slipped: 96,
  slipped_rate: 0.1498,
  slipped_conversations: 33,
  tool_calls: 1676,
  calls_per_turn: 1.0,
  shell_calls: 0,
  shell_conversations: 0,
  index_read_conversations: 0,
  bm25_calls: 436,
  dense_calls: 0,
  grep_calls: 15,
  lookups_before_first_action: 3.27,
  bare_discoverable_calls: 29,
  bare_discoverable_conversations: 9,
  account_email_calls: 713,
  account_email_conversations: 19,
  tool_errors: 26,
  sandbox_failures: 0,
  shell_blocked: 0,
  dense_errors: 0,
  required_docs_read_share: 0.5797,
};

describe('harness health', () => {
  const rows = Object.fromEntries(healthRows(V1_TRAIN, true).map((r) => [r.key, r.value]));

  it('states each count with what it is out of', () => {
    expect(rows.slipped).toBe('96 of 641 replies (15.0%), in 33 of 60 conversations');
    expect(rows.shell).toBe('0 of 60 conversations, 0 commands');
    expect(rows.index).toBe('0 of 60 conversations');
    expect(rows.bare).toBe('29 in 9 of 60 conversations');
    expect(rows.email).toBe('713 in 19 of 60 conversations');
    expect(rows.search).toBe('BM25 436 · dense 0 · grep 15, of 1,676 tool calls');
    expect(rows.lookups).toBe('3.27 a conversation, mean of 60');
    expect(rows.docs).toMatch(/^58% of a task's required documents/);
  });

  it('shows the knowledge rows only where the run had knowledge tools', () => {
    expect(usesKnowledge(V1_TRAIN, null)).toBe(true);
    const airline = { ...V1_TRAIN, bm25_calls: 0, grep_calls: 0, required_docs_read_share: null };
    expect(usesKnowledge(airline, null)).toBe(false);
    expect(healthRows(airline, false).map((r) => r.key)).toEqual(['slipped', 'calls', 'email', 'errors']);
    expect(healthRows({ ...V1_TRAIN, required_docs_read_share: null }, true).find((r) => r.key === 'docs')?.value).toBe('—');
  });

  it('heads the section with a claim and its denominator', () => {
    expect(healthClaim(V1_TRAIN)).toBe('96 of 641 replies were slipped tool calls');
    expect(healthClaim({ ...V1_TRAIN, slipped: 0 })).toBe('none of 641 replies was a slipped tool call');
  });
});
