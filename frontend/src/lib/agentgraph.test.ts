/**
 * The Agent tab's arithmetic, pinned: which messages are model calls, how a tool
 * result finds its call, what each check says, and where an old `?node=` lands.
 * The fixture is the shape `GET /api/runs/{id}/{task}/{trial}` returns.
 */
import { describe, expect, it } from 'vitest';
import type { RewardInfo, ToolSpec, TraceMessage, TrialPayload } from './api';
import { LEGACY_NODES, agentSteps, apiBehind, checks, firstWrite, legacyNode, nodeOfMessage, toolCalls, userSteps, writesBefore } from './agentgraph';

const msg = (i: number, role: TraceMessage['role'], extra: Partial<TraceMessage> = {}): TraceMessage => ({
  i,
  role,
  content: null,
  tool_calls: [],
  id: null,
  requestor: null,
  turn_idx: i,
  usage: role === 'assistant' || role === 'user' ? { prompt_tokens: 100 * i, completion_tokens: 10 } : null,
  seconds: role === 'assistant' ? 1.5 : null,
  error: false,
  ...extra,
});

// greeting · customer · agent calls two tools · two results · agent replies · customer uses their phone · result
const MSGS: TraceMessage[] = [
  msg(0, 'assistant', { content: 'Hi! How can I help you today?', usage: null, seconds: null }),
  msg(1, 'user', { content: 'Change my flight.' }),
  msg(2, 'assistant', {
    tool_calls: [
      { id: 'a', name: 'get_user_details', arguments: { user_id: 'u1' }, requestor: 'assistant' },
      { id: 'b', name: 'get_reservation_details', arguments: { reservation_id: 'R1' }, requestor: 'assistant' },
    ],
  }),
  msg(3, 'tool', { id: 'a', content: '{"user_id":"u1"}', requestor: 'assistant' }),
  msg(4, 'tool', { id: 'b', content: 'Error: not found', requestor: 'assistant', error: true }),
  msg(5, 'assistant', { content: 'I could not find R1.' }),
  msg(6, 'user', { tool_calls: [{ id: 'c', name: 'toggle_airplane_mode', arguments: {}, requestor: 'user' }] }),
  msg(7, 'tool', { id: 'c', content: 'Airplane mode off', requestor: 'user' }),
];

describe('one conversation, read as nodes', () => {
  it("τ²'s fixed greeting is not a model call", () => {
    const st = agentSteps(MSGS);
    expect(st.map((s) => s.i)).toEqual([2, 5]);
    // the first call's new input is the customer's message; the second's, the two results
    expect(st[0].inputs.map((m) => m.i)).toEqual([1]);
    expect(st[1].inputs.map((m) => m.i)).toEqual([3, 4]);
  });

  it('a customer turn answers the last thing the agent said', () => {
    expect(userSteps(MSGS).map((s) => [s.i, s.inputs.map((m) => m.i)])).toEqual([
      [1, [0]],
      [6, [5]],
    ]);
  });

  it('every result pairs with exactly the call that asked for it, by id', () => {
    const calls = toolCalls(MSGS);
    expect(calls.map((c) => [c.call.name, c.by, c.k, c.result?.i])).toEqual([
      ['get_user_details', 'agent', 0, 3],
      ['get_reservation_details', 'agent', 1, 4],
      ['toggle_airplane_mode', 'user', 0, 7],
    ]);
    expect(calls[1].result?.error).toBe(true);
  });

  it("a replay row selects the node it belongs to, the customer's tools apart from the agent's", () => {
    expect(MSGS.map((m) => nodeOfMessage(MSGS, m))).toEqual([
      'agent',
      'user',
      'tool:get_user_details',
      'tool:get_user_details',
      'tool:get_reservation_details',
      'agent',
      'utool:toggle_airplane_mode',
      'utool:toggle_airplane_mode',
    ]);
  });
});

describe('the five checks', () => {
  const ri: RewardInfo = {
    reward: 0,
    reward_basis: ['DB', 'COMMUNICATE'],
    reward_breakdown: { DB: 0, COMMUNICATE: 1 },
    db_check: { db_match: false, db_reward: 0 },
    action_checks: [
      { action: { action_id: '22_0', name: 'update_reservation_flights', arguments: {} }, action_match: false },
      { action: { action_id: '22_1', name: 'get_user_details', arguments: {} }, action_match: true },
    ],
    communicate_checks: [],
    nl_assertions: null,
    env_assertions: [],
  };

  it('colours a check by the basis, and dashes the ones the basis ignores', () => {
    const c = checks(ri, ri.reward_basis!);
    expect(c.map((x) => [x.id, x.cls, x.sub])).toEqual([
      ['check:db', 'err', '✕ differs'],
      ['check:action', 'ext', '1/2 · off'],
      ['check:communicate', 'ok', 'none'],
      ['check:nl', 'ext', 'not run · off'],
      ['check:env', 'ext', 'none · off'],
    ]);
  });
});

describe("the figure this tab drew before s05", () => {
  it('every old box still opens a node of the new graph', () => {
    for (const [old, now] of Object.entries(LEGACY_NODES)) {
      expect(legacyNode(old)).toBe(now);
      expect(['user', 'agent', 'model', 'check:nl']).toContain(now);
    }
  });

  it('leaves a new key alone', () => {
    expect(legacyNode('tool:book_reservation')).toBe('tool:book_reservation');
    expect(legacyNode(null)).toBeNull();
  });
});

describe('an API older than the page', () => {
  it('is a conversation with events and no messages; the tab explains instead of crashing', () => {
    expect(apiBehind({ task_id: '22', events: [] })).toBe(true);
    expect(apiBehind({ task_id: '22', events: [], messages: [] })).toBe(false);
    // nothing loaded yet is not a stale API
    expect(apiBehind(undefined)).toBe(false);
  });
});

describe("where the playground runs a task's expected call", () => {
  const tool = (name: string, mutates: boolean): ToolSpec => ({ name, description: null, type: mutates ? 'write' : 'read', mutates });
  const trial = (messages: TraceMessage[], tools: ToolSpec[], userTools: ToolSpec[] = []) =>
    ({ messages, tools, user_tools: userTools }) as unknown as TrialPayload;
  const READS = [tool('get_user_details', false), tool('get_reservation_details', false)];

  it("is before the first write by either side: the database τ²'s grader starts from", () => {
    // the agent only reads; the customer's airplane-mode toggle at 6 is the first write
    const t = trial(MSGS, READS, [tool('toggle_airplane_mode', true)]);
    expect(firstWrite(t)).toBe(6);
    expect(writesBefore(t, 6)).toHaveLength(0);
    expect(writesBefore(t, 6, 1)).toHaveLength(1);
    expect(writesBefore(t, MSGS.length)).toHaveLength(1);
  });

  it('is the end when nothing was written, which is the same database', () => {
    expect(firstWrite(trial(MSGS, READS, [tool('toggle_airplane_mode', false)]))).toBe(MSGS.length);
  });

  it('counts a write inside a message only once the replay passes it', () => {
    // airline/24's shape: a read and the booking in one message, the booking second
    const msgs = [
      msg(0, 'assistant', { usage: null }),
      msg(1, 'assistant', {
        tool_calls: [
          { id: 'r', name: 'get_user_details', arguments: {}, requestor: 'assistant' },
          { id: 'w', name: 'book_reservation', arguments: {}, requestor: 'assistant' },
        ],
      }),
      msg(2, 'tool', { id: 'r', content: '{}' }),
      msg(3, 'tool', { id: 'w', content: '{}' }),
    ];
    const t = trial(msgs, [tool('get_user_details', false), tool('book_reservation', true)]);
    expect(firstWrite(t)).toBe(1);
    expect(writesBefore(t, 1, 1)).toHaveLength(0);
    expect(writesBefore(t, 1, 2)).toHaveLength(1);
  });
});
