import { describe, expect, it } from 'vitest';
import type { AgentInfo, Registries } from './api';
import { agentVersion, modelFamily, scopeFromLocation, scopeHref, type Scope } from './scope';

const v = (name: string, model: string): AgentInfo =>
  ({ domain: 'airline', name, ref: `airline/${name}`, fingerprint: '', config: { model }, has_helper: false, helper_functions: [], diagnosis: null, runs: [] }) as AgentInfo;
const VERSIONS = [v('v0', 'haiku'), v('v3', 'sonnet'), v('v5', 'opus'), v('v6', 'sonnet')];
const REG = { airline: { champion: { agent: 'v3' } } } as unknown as Registries;

describe('the scope bar', () => {
  it('reads the dataset and the agent back from every tab’s own address', () => {
    expect(scopeFromLocation('/optimise/retail/judge', '?model=sonnet')).toEqual({ dataset: 'retail', agent: 'judge', model: 'sonnet' });
    expect(scopeFromLocation('/optimise/airline', '')).toEqual({ dataset: 'airline', agent: 'answering' });
    expect(scopeFromLocation('/agent/telecom/judge', '')).toMatchObject({ dataset: 'telecom', agent: 'judge' });
    expect(scopeFromLocation('/evals/airline/judge', '?model=opus')).toEqual({ dataset: 'airline', agent: 'judge', model: 'opus' });
    expect(scopeFromLocation('/evals/retail/12', '')).toEqual({ dataset: 'retail', agent: 'answering' });
    expect(scopeFromLocation('/review/golden/airline/20260928T075613Z_airline_v4_train/44/t1/34', '')).toMatchObject({ dataset: 'airline', agent: 'judge' });
    expect(scopeFromLocation('/runs', '?domain=airline&agent=judge')).toMatchObject({ dataset: 'airline', agent: 'judge' });
    // a run's own page names its domain in the run id, and is always the answering agent's
    expect(scopeFromLocation('/runs/20260928T075613Z_airline_v4_train/44/t1', '')).toEqual({ dataset: 'airline', agent: 'answering' });
    // Overview and Rubric say nothing, so the remembered scope stands
    expect(scopeFromLocation('/', '')).toEqual({});
    expect(scopeFromLocation('/rubric', '?model=opus')).toEqual({});
  });

  it('every tab’s scoped address reads back as the scope it was built from', () => {
    const scopes: Scope[] = [
      { dataset: 'airline', agent: 'judge', model: 'sonnet' },
      { dataset: 'retail', agent: 'answering', model: '' },
      { dataset: 'banking_knowledge', agent: 'answering', model: 'haiku' },
    ];
    for (const s of scopes) {
      for (const tab of ['evals', 'runs', 'optimise', 'review'] as const) {
        const url = scopeHref(tab, s);
        const [path, query = ''] = url.split('?');
        const back = scopeFromLocation(path, query ? `?${query}` : '');
        expect(back.dataset).toBe(s.dataset);
        expect(back.agent).toBe(s.agent);
        // the model rides in the address wherever it filters: not on the golden answers, nor the answering agent's tasks
        const carries = !(tab === 'review' && s.agent === 'judge') && !(tab === 'evals' && s.agent === 'answering');
        if (s.model && carries) expect(back.model).toBe(s.model);
      }
    }
  });

  it('the Agent tab opens the champion when it fits the model, else the newest version that does', () => {
    const s: Scope = { dataset: 'airline', agent: 'answering', model: '' };
    expect(agentVersion(s, VERSIONS, REG)).toBe('v3');
    expect(agentVersion({ ...s, model: 'opus' }, VERSIONS, REG)).toBe('v5');
    expect(agentVersion({ ...s, model: 'haiku' }, VERSIONS, REG)).toBe('v0');
    expect(scopeHref('agent', { ...s, model: 'opus' }, { versions: VERSIONS, registry: REG })).toBe('/agent/airline/v5');
    expect(scopeHref('agent', { ...s, agent: 'judge' })).toBe('/agent/airline/judge');
  });

  it('every model string here falls in one family', () => {
    expect(['claude-sdk/claude-haiku-4-5', 'claude-sonnet-5', 'opus', 'claude-opus-5-5'].map(modelFamily)).toEqual(['haiku', 'sonnet', 'opus', 'opus']);
  });
});
