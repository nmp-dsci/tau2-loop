/**
 * workflow_rag's version view on the Agent tab, pinned: each rN's parent, which layers differ
 * from it (the seed counted as "its parent's library" or not), the figure's marks, a session's
 * counts and the table. The fixtures are the shape `GET /api/workflows/<d>/rag` returns for
 * banking's r1, r3, r4, r5 and r13.
 */
import { renderToStaticMarkup } from 'react-dom/server';
import { StaticRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import type { AgentInfo, RagSession, RagVersion } from './api';
import { RagFigure, RagNodePanel, RagVersionTable, fileOrigin, ragChanged, ragClaim, ragLayers, ragParent, readersOf, sessionCounts } from './ragarch';

const KT = ['KB_search_bm25', 'KB_search_dense', 'shell'].map((name) => ({ name, description: `${name}.`, type: 'read', mutates: false }));
const DOC = { name: 'tool_description', description: 'The whole description of one tool.', type: 'read', mutates: false, harness: true };
const rv = (name: string, extra: Partial<RagVersion>): RagVersion => ({
  name,
  model: 'opus',
  effort: 'high',
  retrieval: 'alltools_minilm',
  tool_mode: 'native_parallel',
  max_steps: 30,
  merge_steps: 12,
  result_chars: 20000,
  prompt: `${name}'s research prompt`,
  decide_prompt: '',
  merge_prompt: '',
  tools: KT,
  seed: null,
  merge: null,
  workers: 1,
  order: 'id',
  order_seed: 0,
  queue: '',
  keep_tools: false,
  merge_write: 'full',
  rubric: [],
  tool_docs: false,
  describe_named: false,
  based_on: null,
  changes: null,
  files: { 'rag_agent.md': { sha: `${name}-research`, chars: 5000 } },
  library: { workflows: 55, commits: null },
  ...extra,
});
const concurrent = { merge: 'concurrent' as const, workers: 4, order: 'shuffled', order_seed: 21, queue: 'after_seed' };
const R1 = rv('r1', { library: { workflows: 64, commits: null } });
const R3 = rv('r3', {
  ...concurrent,
  seed: 'r2',
  decide_prompt: 'decide which jobs',
  files: { 'rag_agent.md': { sha: 'r3-research', chars: 8238 }, 'decide.md': { sha: 'decide', chars: 885 }, 'merge.md': { sha: 'r3-merge', chars: 1986 } },
  library: { workflows: 55, commits: 151 },
});
// r4: r3's research and decide bytes, its own merge prompt, merges written as edits; never run
const R4 = rv('r4', {
  ...concurrent,
  seed: 'r3',
  keep_tools: true,
  merge_write: 'edits',
  files: { 'rag_agent.md': { sha: 'r3-research', chars: 8238 }, 'decide.md': { sha: 'decide', chars: 885 }, 'merge.md': { sha: 'r4-merge', chars: 3720 } },
  library: { workflows: 55, commits: 0 },
});
// r5: based on r4 but seeded with r3's library (r4 never ran), a new research rule and D10
const R5 = rv('r5', {
  ...concurrent,
  seed: 'r3',
  based_on: 'r4',
  keep_tools: true,
  merge_write: 'edits',
  rubric: ['D10'],
  changes: { version: 'r5', based_on: 'r4', seed: 'r3', defect: 'D1', why: 'D1 (closing a frozen card)', changes: [{ file: 'rag_agent.md', what: 'reading rule 1', why: 'a general rule' }] },
  files: { 'rag_agent.md': { sha: 'r5-research', chars: 9418 }, 'decide.md': { sha: 'decide', chars: 885 }, 'merge.md': { sha: 'r4-merge', chars: 3720 } },
  library: { workflows: 55, commits: 7 },
});
const R13 = rv('r13', { ...R5, name: 'r13', based_on: 'r5', seed: 'r5', tool_docs: true, tools: [...KT, DOC], changes: null });
const ALL = [R1, R3, R4, R5, R13];
const agent = (name: string, workflows: string | null) => ({ domain: 'banking_knowledge', name, config: { workflows } }) as unknown as AgentInfo;
const AGENTS = [agent('v6', 'r1'), agent('v7', 'r1'), agent('v8', 'r3'), agent('v4', null)];

describe('an rN against its parent', () => {
  it('r1 has no parent; r4 is measured against its seed; r5 against its change record', () => {
    expect(ragParent(R1, ALL)).toBeNull();
    expect(ragParent(R4, ALL)?.name).toBe('r3');
    expect(ragParent(R5, ALL)?.name).toBe('r4');
    expect(ragClaim(R1, ALL)).toBe('r1 is the first version: nothing to compare');
  });

  it("a prompt copied byte for byte reads as the version that wrote it", () => {
    expect(fileOrigin(R4, 'rag_agent.md', ALL)).toBe('r3');
    expect(fileOrigin(R4, 'merge.md', ALL)).toBe('r4');
    expect(ragLayers(R4, ALL).find((l) => l.key === 'decide')?.value).toBe("r3's, byte for byte");
    expect(ragLayers(R1, ALL).find((l) => l.key === 'decide')?.value).toBe('none');
  });

  it("starting from the parent's library is not a change; r5 starting from r3's, not r4's, is", () => {
    expect([...ragChanged(R4, R3, ALL)].sort()).toEqual(['merge_md', 'merge_write']);
    expect([...ragChanged(R5, R4, ALL)].sort()).toEqual(['research', 'rubric', 'seed']);
    expect(ragLayers(R5, ALL).find((l) => l.key === 'seed')?.detail).toContain("not its parent's (r4)");
    expect(ragClaim(R5, ALL)).toBe('r5 differs from r4 in 3 of 12 layers');
  });

  it('r13 adds tool descriptions: the layer and the tool are marked, nothing else', () => {
    expect([...ragChanged(R13, R5, ALL)]).toEqual(['tool_docs']);
  });

  it('the answering versions that read a library are the ones whose workflow tools name it', () => {
    expect(readersOf(R1, AGENTS).map((a) => a.name)).toEqual(['v6', 'v7']);
    expect(readersOf(R4, AGENTS)).toEqual([]);
  });
});

describe('the figure, drawn from the folder', () => {
  const html = (v: RagVersion, s: RagSession | null = null) =>
    renderToStaticMarkup(<RagFigure domain="banking_knowledge" v={v} all={ALL} readers={readersOf(v, AGENTS)} s={s} node={null} onPick={() => {}} />);

  it("marks r13's tool_description and its layer against r5, and nothing for r1", () => {
    const out = html(R13);
    expect(out).toContain('tool_description · harness');
    expect(out.match(/>new</g)?.length).toBe(1);
    expect(html(R1)).not.toMatch(/class="nd hi/);
    expect(html(R1)).toContain('none: no merge to check');
  });

  it("draws r4's three prompts with their origins, its library and who reads it", () => {
    const out = html(R4);
    expect(out).toContain("rag_agent.md · r3&#x27;s");
    expect(out).toContain('merge.md · its own');
    expect(out).toContain('55 workflows · 0 commits');
    expect(out).toContain('no answering version yet');
    expect(html(R3)).toContain('v8');
  });

  const S: RagSession = {
    meta: { id: 's', domain: 'banking_knowledge', agent: 'r5', task_id: 'task_016', model: 'opus', effort: 'high', retrieval: 'alltools_minilm', started: '20261011T000000Z', status: 'done', steps: 4, tool_calls: 3, input_tokens: 1, output_tokens: 1, duration_ms: 1, error: null },
    events: [
      { t: 0, kind: 'start', question: 'the script' },
      { t: 1, kind: 'model', step: 1 },
      { t: 2, kind: 'tool', name: 'KB_search_bm25', args: { query: 'close card' } },
      { t: 3, kind: 'tool', name: 'shell', args: { command: 'ls' } },
      { t: 4, kind: 'model', step: 2 },
      { t: 5, kind: 'merge' },
      { t: 6, kind: 'model', step: 3 },
      { t: 7, kind: 'decided', decision: { closing_a_card: ['closing_a_card'] } },
      { t: 8, kind: 'locked', waited_ms: 1200, jobs: ['closing_a_card'] },
      { t: 9, kind: 'note', text: 'The merge failed its checks; sent back once.' },
      { t: 10, kind: 'tool', name: 'KB_search_bm25', args: { query: 'freeze' } },
      { t: 11, kind: 'model', step: 4 },
      { t: 12, kind: 'committed', versions: { closing_a_card: 3 } },
      { t: 13, kind: 'merged', taken: ['closing_a_card'], rejected: [] },
    ],
    output: null,
    score: null,
  };

  it("counts a session's research and merge turns, its calls, its lock and its gate", () => {
    const c = sessionCounts(S);
    expect([c.research, c.mergeTurns, c.decided, c.waitedMs]).toEqual([2, 2, 1, 1200]);
    expect(c.calls.get('KB_search_bm25')?.length).toBe(2);
    expect(c.sentBack.length).toBe(1);
    const out = html(R5, S);
    expect(out).toContain('2 turns');
    expect(out).toContain('×2');
    expect(out).toContain('✓ on resend');
    expect(out).toContain('+1');
  });

  it("draws no counts from another version's session", () => {
    expect(html(R4, S)).not.toContain('×2');
  });

  it("opens a prompt's text and what the version changed in it", () => {
    const out = renderToStaticMarkup(
      <RagNodePanel domain="banking_knowledge" node="file:rag_agent.md" v={R5} all={ALL} readers={[]} hist={null} s={null} onVersion={() => {}} onClose={() => {}} />,
    );
    expect(out).toContain('reading rule 1');
    expect(out).toContain('r5&#x27;s research prompt');
    expect(out).toContain('Differs from r4');
  });
});

describe('the comparison table', () => {
  const out = renderToStaticMarkup(
    <StaticRouter location="/agent/banking_knowledge/workflow_rag">
      <RagVersionTable versions={ALL} readers={(v) => readersOf(v, AGENTS)} hist={null} sessions={() => 0} current="r5" onVersion={() => {}} />
    </StaticRouter>,
  );

  it("marks each version's changed cells against its own parent", () => {
    // r3 against r2 is not drawn (r2 is not in the fixture); r4's 2, r5's 3, r13's 1
    expect(out.match(/class="wrap chg"/g)?.length).toBe(6);
  });

  it('says why a version was made, or that it has no change record', () => {
    expect(out).toContain('D1 · rag_agent.md');
    expect(out).toContain('no change record');
  });
});
