/**
 * Evals › a banking task's required documents, pinned on task_001 as
 * `GET /api/domains/banking_knowledge/tasks/task_001/reads` returns it: each cell is read, seen or
 * never reached, and every count carries its denominator.
 */
import { describe, expect, it } from 'vitest';
import { docState, readTally } from './docreads';

const GOLD = 'doc_credit_cards_gold_rewards_card_001';
const SILVER = 'doc_credit_cards_silver_rewards_card_001';
const BRONZE = 'doc_credit_cards_bronze_rewards_card_001';
const PLATINUM = 'doc_credit_cards_platinum_rewards_card_001';
const REQUIRED = [GOLD, SILVER, BRONZE, PLATINUM];

const V1 = { read: [SILVER], seen: [] };
const V2 = { read: [BRONZE, GOLD], seen: [] };
const V3 = { read: [], seen: [BRONZE, GOLD, PLATINUM, SILVER] };

describe('required documents, per version', () => {
  it('marks each cell read, seen or never reached', () => {
    expect(REQUIRED.map((id) => docState(V1, id))).toEqual(['none', 'read', 'none', 'none']);
    expect(REQUIRED.map((id) => docState(V2, id))).toEqual(['read', 'none', 'read', 'none']);
    expect(REQUIRED.map((id) => docState(V3, id))).toEqual(['seen', 'seen', 'seen', 'seen']);
  });

  it('a document read whole is never also counted as seen', () => {
    expect(docState({ read: [GOLD], seen: [GOLD] }, GOLD)).toBe('read');
    expect(readTally({ read: [GOLD], seen: [GOLD, SILVER] }, REQUIRED)).toEqual({ read: '1 / 4', seen: '1 / 4' });
  });

  it('counts only the required documents, with the denominator', () => {
    expect(readTally(V1, REQUIRED)).toEqual({ read: '1 / 4', seen: '0 / 4' });
    expect(readTally(V2, REQUIRED)).toEqual({ read: '2 / 4', seen: '0 / 4' });
    expect(readTally(V3, REQUIRED)).toEqual({ read: '0 / 4', seen: '4 / 4' });
    expect(readTally({ read: ['doc_other_001'], seen: [] }, REQUIRED).read).toBe('0 / 4');
  });
});
