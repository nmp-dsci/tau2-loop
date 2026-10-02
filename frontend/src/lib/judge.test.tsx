import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { EventList } from '../pages/Trace';
import { type JudgeView, judgeMarks } from './judge';

const EVENTS = [
  { type: 'assistant', i: 6, text: 'To downgrade the cabin I need your reservation id.' },
  { type: 'user', i: 7, text: 'It is ABC123.' },
  { type: 'tool_call', i: 9, by: 'agent', name: 'update_reservation_flights', arguments: { reservation_id: 'ABC123' } },
];

const VIEW = {
  labels: {
    key: 'r/29/t1',
    run: 'r',
    version: 'v0',
    task: '29',
    passed: false,
    mode: 'wrong_write',
    fold: 1,
    half: 'read',
    suspect: false,
    first_wrong: null,
    checkpoints: [
      { msg: 6, kind: 'reply', judge: 'plan', trigger: true, live: true, label: 'unlabelled', pinned: false },
      { msg: 9, kind: 'write', judge: 'call', trigger: false, live: false, label: 'bad', pinned: true },
    ],
  },
  gold: {
    key: 'r/29/t1',
    annotator: { model: 'claude-opus-5-5', effort: 'high' },
    answer: {
      summary: '',
      first_wrong_step: null,
      checkpoints: [
        { id: 'c1', msg: 6, kind: 'reply', verdict: 'allow', check: null, rule: null, evidence: [], detectable: null, why: 'a normal step', fix: null },
      ],
    },
  },
  verdicts: {
    replay_id: 'x',
    judge: 'airline/plan/j1',
    model: 'claude-sonnet-5',
    threshold: 0.5,
    finished: true,
    items: [
      {
        msg: 6,
        kind: 'reply',
        verdict: 'block',
        note: null,
        error: null,
        raw: { is_plan: false, verdict: 'block', confidence: 0.85, check: 2, rule: 'transcript', evidence: [], fix: 'ask for the id', why: '' },
      },
    ],
  },
} as JudgeView;

/** Each top-level `.ev` block of the rendered list, in order. */
function blocks(html: string): string[] {
  return html.split(/(?=<div class="ev )/).filter((b) => b.startsWith('<div class="ev '));
}

describe('the LLM judge on a conversation', () => {
  it('speaks in a bar of its own after the message it reviews, never inside the answering agent’s', () => {
    const bs = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(VIEW)} />));
    expect(bs.map((b) => /^<div class="ev ([\w ]+)"/.exec(b)?.[1])).toEqual(['text', 'judge', 'user', 'tool_use', 'judge']);
    expect(bs[0]).toContain('answering agent');
    expect(bs[0]).not.toContain('verdict');
    expect(bs[1]).toContain('LLM judge · plan judge j1');
    expect(bs[1]).toContain('on message 6');
  });

  it('leads with its verdict, then the golden answer it is scored against, then J0’s label', () => {
    const bar = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(VIEW)} />))[1];
    const at = (s: string) => bar.indexOf(s);
    expect(at('verdict')).toBeGreaterThan(-1);
    expect(at('verdict')).toBeLessThan(at('golden answer'));
    expect(at('golden answer')).toBeLessThan(at('J0 label'));
    expect(bar).toContain('differs from');
  });

  it('says when it was not asked, as at a write the plan judge never sees', () => {
    const bar = blocks(renderToStaticMarkup(<EventList events={EVENTS} marks={judgeMarks(VIEW)} />))[4];
    expect(bar).toContain('LLM judge · not asked at this checkpoint');
    expect(bar).not.toContain('>verdict<');
  });
});
