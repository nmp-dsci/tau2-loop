import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { EventList } from '../pages/Trace';
import { type JudgeView, judgeMarks } from './judge';

const EVENTS = [
  { type: 'assistant', i: 6, text: 'To downgrade the cabin I need your reservation id. Shall I proceed?' },
  { type: 'user', i: 7, text: 'It is ABC123.' },
  { type: 'tool_call', i: 9, by: 'agent', name: 'update_reservation_flights', arguments: { reservation_id: 'ABC123' } },
  { type: 'tool_call', i: 11, by: 'agent', name: 'transfer_to_human_agents', arguments: { summary: 'the user wants a refund' } },
];

const VIEW = {
  labels: {
    key: 'r/29/t1',
    run: 'r',
    version: 'v3',
    task: '29',
    passed: false,
    mode: 'wrong_write',
    fold: 1,
    half: 'read',
    suspect: false,
    first_wrong: null,
    checkpoints: [
      { msg: 6, kind: 'reply', judge: 'plan', trigger: true, live: true, judged: false, label: 'unlabelled', pinned: false },
      { msg: 9, kind: 'write', judge: 'call', trigger: false, live: false, judged: true, label: 'bad', pinned: true },
      { msg: 11, kind: 'transfer', judge: 'direction', trigger: false, live: false, judged: true, label: 'unlabelled', pinned: false },
    ],
  },
  gold: {
    key: 'r/29/t1',
    annotator: { model: 'claude-opus-5-5', effort: 'high' },
    answer: {
      summary: '',
      checkpoints: [
        { id: 'c1', msg: 6, kind: 'reply', verdict: 'block', check: 2, rule: 'transcript', evidence: [], detectable: true, why: 'a stray remark', fix: null },
        { id: 'c2', msg: 9, kind: 'write', verdict: 'block', check: 3, rule: 'transcript', evidence: [], detectable: true, why: 'the wrong cabin', fix: null },
        { id: 'c3', msg: 11, kind: 'transfer', verdict: 'allow', check: null, rule: null, evidence: [], detectable: null, why: 'policy allows it', fix: null },
      ],
      first_wrong_step: { msg: 9, kind: 'write', blame: 'agent', why: 'the wrong cabin' },
    },
  },
  verdicts: {
    replay_id: 'x',
    judge: 'airline/call/j1',
    model: 'claude-sonnet-5',
    threshold: 0.5,
    finished: true,
    items: [
      {
        msg: 9,
        kind: 'write',
        verdict: 'block',
        note: null,
        error: null,
        raw: { is_plan: false, verdict: 'block', confidence: 0.85, check: 3, rule: 'transcript', evidence: [], fix: 'keep economy', why: '' },
      },
    ],
  },
} as JudgeView;

/** Each top-level `.ev` block of the rendered list, in order. */
function blocks(html: string): string[] {
  return html.split(/(?=<div class="ev )/).filter((b) => b.startsWith('<div class="ev '));
}

describe('the LLM judge on a conversation', () => {
  it('speaks only after a write or a transfer, before it runs, never after a text reply', () => {
    const bs = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(VIEW)} />));
    expect(bs.map((b) => /^<div class="ev ([\w ]+)"/.exec(b)?.[1])).toEqual(['text', 'user', 'tool_use', 'judge', 'tool_use', 'judge']);
    expect(bs[3]).toContain('LLM judge · tool-call judge j1, before the call runs');
    expect(bs[3]).toContain('on message 9');
    expect(bs.join('')).not.toContain('on message 6');
  });

  it('leads with its verdict, then the golden answer it is scored against, then J0’s label', () => {
    const bar = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(VIEW)} />))[3];
    const at = (s: string) => bar.indexOf(s);
    expect(at('verdict')).toBeGreaterThan(-1);
    expect(at('verdict')).toBeLessThan(at('golden answer'));
    expect(at('golden answer')).toBeLessThan(at('J0 label'));
    expect(bar).toContain('agrees with');
  });

  it('says when it gave no verdict at a call, and when no tool-call judge has run', () => {
    const bar = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(VIEW)} />))[5];
    expect(bar).toContain('LLM judge · no verdict at this call');
    const none = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks({ ...VIEW, verdicts: null })} />))[3];
    expect(none).toContain('no tool-call judge has run yet');
  });

  it('offers “correct it” on each golden answer only where a person can save one', () => {
    const edit = { domain: 'airline', onSaved: () => {} };
    expect(blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(VIEW, edit)} />))[3]).toContain('>correct it</button>');
    expect(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(VIEW)} />)).not.toContain('correct it');
  });

  it('shows a person’s live correction at once: the verdict flips, the annotator’s stays beside it, and the first wrong step moves', () => {
    const check = {
      id: 1,
      item_id: 'r/29/t1#9',
      conv_key: 'r/29/t1',
      msg: 9,
      verdict: 'correct' as const,
      correction: { verdict: 'allow', first_wrong_msg: 11 },
      note: 'economy was what the user asked for',
      author: '',
      created_at: '2026-10-02 10:00:00+00',
    };
    const bs = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks({ ...VIEW, verdicts: null, checks: { '9': check } })} />));
    expect(bs[3]).toContain('a person’s correction');
    expect(bs[3]).toContain('economy was what the user asked for');
    expect(bs[3]).toContain('the annotator said block');
    expect(bs[3]).toContain('not yet in the gold file');
    expect(bs[3]).not.toContain('the golden first wrong step');
    expect(bs[5]).toContain('the golden first wrong step');
  });

  it('allows every call of a pass by the passed rule, whatever the annotator said, with nothing to correct', () => {
    const pass = { ...VIEW, labels: { ...VIEW.labels, passed: true, mode: 'pass' }, verdicts: null } as JudgeView;
    const edit = { domain: 'airline', onSaved: () => {} };
    const bar = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(pass, edit)} />))[3];
    expect(bar).toContain('the passed rule');
    expect(bar).toContain('the annotator said block · the wrong cabin');
    expect(bar).not.toMatch(/>block</);
    expect(bar).not.toContain('correct it');
  });
});
