/**
 * The Agent tab's version view (s13 §2), pinned: which layers differ from a version's parent,
 * what the figure and the table mark, the trial graph's knowledge tools, and the model panel's
 * tool mode. The fixtures are the shape `GET /api/agents` returns for banking's v1 and v2, and
 * for a v3 on AllTools with native tool calls and the identity note.
 */
import { renderToStaticMarkup } from 'react-dom/server';
import { StaticRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import type { AgentInfo, RunMeta, ToolSpec, TrialPayload, VersionNode } from './api';
import { NodePanel } from './agentgraph';
import { ArchitectureFigure, VersionTable, changed, layers, parentOf, toolModeLong, versionClaim, withKnowledge } from './architecture';

const BM25 = { variant: 'bm25_grep', tools: ['KB_search', 'grep'], dense_model: null, template: 'classic_rag_bm25.md' };
const ALLTOOLS = {
  variant: 'alltools_minilm',
  tools: ['KB_search_bm25', 'KB_search_dense', 'shell'],
  dense_model: 'local:sentence-transformers/all-MiniLM-L6-v2',
  template: 'all_tools.md',
};
const base = (name: string, extra: Partial<AgentInfo>): AgentInfo => ({
  domain: 'banking_knowledge',
  name,
  ref: `banking_knowledge/${name}`,
  fingerprint: `${name}0000000000`,
  config: { model: 'sonnet', effort: 'medium', tool_mode: 'json', max_steps: 200, retrieval: null, identity_note: false },
  has_helper: false,
  helper_functions: [],
  diagnosis: null,
  runs: [],
  retrieval: 'bm25_grep',
  retrieval_info: BM25,
  tool_mode: 'json',
  surfaces_present: ['system.md'],
  surfaces: { 'system.md': { sha: 'ec1b89dbe2b8', chars: 366 } },
  prompt_layers: ['system.md', 'policy: classic_rag_bm25.md', 'CLOCK_NOTE'],
  made_by: { kind: 'model swap', from: 'v0', cycle: null, detail: 'model: haiku → sonnet' },
  parent: null,
  ...extra,
});
const V1 = base('v1', {});
const V2 = base('v2', {
  has_helper: true,
  helper_functions: ['_load_args', 'on_tool_call', 'on_reply'],
  surfaces_present: ['system.md', 'helper.py'],
  surfaces: { 'system.md': { sha: '4db83a622571', chars: 6942 }, 'helper.py': { sha: '2c00bc30f4bf', chars: 4369 } },
  made_by: { kind: 'loop cycle', from: 'v1', cycle: 1, detail: 'classic opus optimiser' },
  parent: 'v1',
});
const V3 = base('v3', {
  config: { model: 'sonnet', effort: 'medium', tool_mode: 'native', max_steps: 200, retrieval: 'alltools_minilm', identity_note: true },
  retrieval: 'alltools_minilm',
  retrieval_info: ALLTOOLS,
  tool_mode: 'native',
  prompt_layers: ['system.md', 'policy: all_tools.md', 'CLOCK_NOTE', 'identity note'],
  made_by: { kind: 'tool change', from: 'v1', cycle: null, detail: 'retrieval: None → alltools_minilm' },
  parent: 'v1',
});
const ALL = [V1, V2, V3];
const tool = (name: string, type: ToolSpec['type'] = 'read'): ToolSpec => ({ name, description: `${name}.`, type, mutates: type === 'write' });
const DOMAIN_TOOLS = [tool('KB_search'), tool('grep'), tool('get_user_information_by_id'), tool('log_verification', 'write')];

describe('a version against its parent', () => {
  it('v1 has no parent left: its source v0 was retired, so nothing is marked', () => {
    expect(parentOf(V1, ALL)).toBeNull();
    expect(changed(V1, null, ALL).size).toBe(0);
    expect(versionClaim(V1, ALL)).toBe('v1 has no parent here: v0 was retired');
  });

  it("v2's loop cycle changed system.md and added helper.py, nothing else", () => {
    expect([...changed(V2, V1, ALL)].sort()).toEqual(['code', 'system']);
    expect(versionClaim(V2, ALL)).toBe('v2 differs from v1 in 2 of 9 layers');
  });

  it("v3 keeps v1's prompt bytes and swaps the tools, the tool mode and the prompt's notes", () => {
    expect([...changed(V3, V1, ALL)].sort()).toEqual(['dense', 'layers', 'retrieval', 'template', 'tool_mode']);
    const sys = layers(V3, ALL).find((l) => l.key === 'system');
    expect(sys?.value).toBe("v1's, byte for byte");
    expect(layers(V3, ALL).find((l) => l.key === 'layers')?.value).toBe('system.md + policy + CLOCK_NOTE + identity note');
  });

  it('a dataset without retrieval is compared on six layers, not nine', () => {
    const a0 = base('v0', { domain: 'airline', retrieval: null, retrieval_info: null, prompt_layers: ['system.md', 'policy: tau2 domain policy', 'CLOCK_NOTE'] });
    expect(layers(a0, [a0]).map((l) => l.key)).toEqual(['model', 'tool_mode', 'layers', 'system', 'code', 'steps']);
  });
});

describe('the figure, drawn from the folder alone', () => {
  const html = (v: AgentInfo) => renderToStaticMarkup(<ArchitectureFigure v={v} all={ALL} tools={DOMAIN_TOOLS} />);

  it("marks only v3's changed boxes, each with the words beside the colour", () => {
    const out = html(V3);
    expect(out).toContain('native tool calls');
    expect(out).toContain('KB_search_dense');
    expect(out).toContain('local:sentence-transformers/all-MiniLM-L6-v2');
    expect(out).toContain('identity note');
    // the knowledge tools v1 never had read `new`; changed boxes read `≠ v1`
    expect(out.match(/>new</g)?.length).toBe(3);
    expect(out).toContain('≠ v1');
    // the dataset's own tools are listed once, and v1's knowledge tools are not drawn for v3
    expect(out).toContain('get_user_information_by_id');
    expect(out).not.toContain('>KB_search<');
  });

  it('marks nothing for a version with no parent', () => {
    expect(html(V1)).not.toMatch(/class="nd hi"/);
  });
});

describe('the comparison table', () => {
  const hist = [
    { version: 'v1', train: { run_id: 'r1', passed: 2, n: 60, cut: 3, model: null, effort: null }, test: { run_id: 't1', passed: 6, n: 37, cut: 3, model: null, effort: null } },
    { version: 'v2', train: { run_id: 'r2', passed: 17, n: 60, cut: 3, model: null, effort: null }, test: null },
  ] as unknown as VersionNode[];
  const out = renderToStaticMarkup(
    <StaticRouter location="/agent/banking_knowledge/v3">
      <VersionTable versions={ALL} hist={hist} current="v3" />
    </StaticRouter>,
  );

  it("marks each version's changed cells against its own parent: v2's two and v3's five", () => {
    expect(out.match(/class="wrap chg"/g)?.length).toBe(7);
    expect(out.match(/≠ v1/g)?.length).toBe(7);
  });

  it('carries every pass count with its denominator, and says when a version has not run', () => {
    expect(out).toContain('2 / 60');
    expect(out).toContain('6 / 37');
    expect(out).toContain('17 / 60');
    expect(out).toContain('not run');
  });
});

describe("the trial graph's knowledge tools", () => {
  const t = { tools: DOMAIN_TOOLS } as unknown as TrialPayload;

  it("are the version's own when its retrieval differs from the extract's", () => {
    expect(withKnowledge(t, ALLTOOLS).tools.map((x) => x.name)).toEqual(['KB_search_bm25', 'KB_search_dense', 'shell', 'get_user_information_by_id', 'log_verification']);
    expect(withKnowledge(t, ALLTOOLS).tools[0].type).toBe('read');
  });

  it("leave the extract alone when they are the extract's", () => {
    expect(withKnowledge(t, BM25)).toBe(t);
    expect(withKnowledge(t, null)).toBe(t);
  });
});

describe("the model panel's tool mode", () => {
  it('says a native version calls the tools, and a JSON version writes them in its reply', () => {
    expect(toolModeLong('native')).toContain('returns to tau2 unrun');
    expect(toolModeLong('json')).toContain('JSON contract');
    const meta = { run_id: 'r', model: 'claude-sdk/claude-sonnet-5', user_model: 'u', judge_model: 'j', sampling: 'cli-default', tool_mode: 'native' } as unknown as RunMeta;
    const tr = { result: { cost_usd_est: 0 }, messages: [] } as unknown as TrialPayload;
    const html = renderToStaticMarkup(
      <StaticRouter location="/agent/banking_knowledge/v3">
        <NodePanel node="model" step={null} t={tr} agent={null} meta={meta} playground={false} pg={null} onGo={() => {}} onRun={() => {}} onClose={() => {}} />
      </StaticRouter>,
    );
    expect(html).toContain('the model is given the tools and calls them');
    expect(html).not.toContain('not as native tool calls');
  });
});
