import { describe, expect, it } from 'vitest';
import type { AgentInfo, Registries } from './api';
import { agentVersion, expApplies, experimentsOf, modelFamily, scopeFromLocation, scopeHref, taskApplies, type Scope } from './scope';

const v = (name: string, model: string, domain = 'airline'): AgentInfo =>
  ({ domain, name, ref: `${domain}/${name}`, fingerprint: '', config: { model }, has_helper: false, helper_functions: [], diagnosis: null, runs: [] }) as AgentInfo;
const VERSIONS = [v('v0', 'haiku'), v('v3', 'sonnet'), v('v5', 'opus'), v('v6', 'sonnet'), v('v1', 'haiku', 'telecom')];
const REG = { airline: { champion: { agent: 'v3' } } } as unknown as Registries;

describe('the scope bar', () => {
  it('reads the dataset, the agent and the experiment back from every tab’s own address', () => {
    expect(scopeFromLocation('/optimise/retail', '?exp=v2')).toEqual({ dataset: 'retail', agent: 'answering', exp: 'v2' });
    // an open task in the answering agent's Evals is the scope's task; its list is none
    expect(scopeFromLocation('/evals/airline/39', '')).toEqual({ dataset: 'airline', agent: 'answering', task: '39' });
    expect(scopeFromLocation('/evals/airline', '')).toEqual({ dataset: 'airline', agent: 'answering', task: '' });
    expect(scopeFromLocation('/runs', '?domain=airline&exp=v6&task=39')).toEqual({ dataset: 'airline', agent: 'answering', exp: 'v6', task: '39' });
    expect(scopeFromLocation('/evals/airline/judge', '?task=6')).toMatchObject({ agent: 'judge', task: '6' });
    expect(scopeFromLocation('/optimise/airline', '')).toEqual({ dataset: 'airline', agent: 'answering' });
    expect(scopeFromLocation('/agent/telecom/judge', '')).toMatchObject({ dataset: 'telecom', agent: 'judge' });
    expect(scopeFromLocation('/evals/airline/judge', '?exp=v6')).toEqual({ dataset: 'airline', agent: 'judge', exp: 'v6' });
    expect(scopeFromLocation('/evals/retail/12', '')).toEqual({ dataset: 'retail', agent: 'answering', task: '12' });
    expect(scopeFromLocation('/review/golden/airline/20260928T075613Z_airline_v4_train/44/t1/34', '')).toMatchObject({ dataset: 'airline', agent: 'judge' });
    expect(scopeFromLocation('/runs', '?domain=airline&agent=judge')).toMatchObject({ dataset: 'airline', agent: 'judge' });
    expect(scopeFromLocation('/runs', '?domain=airline&exp=v6')).toEqual({ dataset: 'airline', agent: 'answering', exp: 'v6' });
    // a run's own page names its domain in the run id, and is always the answering agent's
    expect(scopeFromLocation('/runs/20260928T075613Z_airline_v4_train/44/t1', '')).toEqual({ dataset: 'airline', agent: 'answering' });
    // Overview and Rubric say nothing, so the remembered scope stands
    expect(scopeFromLocation('/', '')).toEqual({});
    expect(scopeFromLocation('/rubric', '?exp=v6')).toEqual({});
  });

  it('every tab’s scoped address reads back as the scope it was built from', () => {
    const scopes: Scope[] = [
      { dataset: 'airline', agent: 'judge', exp: 'v6', task: '6' },
      { dataset: 'retail', agent: 'answering', exp: '', task: '' },
      { dataset: 'airline', agent: 'answering', exp: 'v4', task: '39' },
      { dataset: 'telecom', agent: 'answering', exp: '', task: '[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]' },
    ];
    for (const s of scopes) {
      for (const tab of ['evals', 'runs', 'optimise', 'review'] as const) {
        const url = scopeHref(tab, s);
        const [path, query = ''] = url.split('?');
        const back = scopeFromLocation(path, query ? `?${query}` : '');
        expect(back.dataset).toBe(s.dataset);
        expect(back.agent).toBe(s.agent);
        // the experiment rides in the address wherever it filters
        if (s.exp && expApplies(tab, s.agent)) expect(back.exp).toBe(s.exp);
        if (s.task && taskApplies(tab, s.agent)) expect(back.task).toBe(s.task);
      }
    }
  });

  it('applies an experiment only where a version of the answering agent made the rows', () => {
    expect(['evals', 'runs', 'optimise', 'agent', 'review'].filter((t) => expApplies(t as never, 'answering'))).toEqual(['runs', 'optimise', 'agent', 'review']);
    expect(['evals', 'runs', 'optimise', 'agent', 'review'].filter((t) => expApplies(t as never, 'judge'))).toEqual(['evals', 'review']);
  });

  it('applies a task where the rows are tasks or their conversations', () => {
    expect(['evals', 'runs', 'optimise', 'agent', 'review'].filter((t) => taskApplies(t as never, 'answering'))).toEqual(['evals', 'runs', 'agent', 'review']);
    expect(['evals', 'runs', 'optimise', 'agent', 'review'].filter((t) => taskApplies(t as never, 'judge'))).toEqual(['evals', 'review']);
  });

  it('lists a dataset’s experiments from its versions and its runs, in order', () => {
    const runs = [
      { domain: 'airline', agent: 'v6', dry_run: false },
      { domain: 'airline', agent: 'v10', dry_run: false },
      { domain: 'airline', agent: 'v9', dry_run: true },
      { domain: 'telecom', agent: 'v2', dry_run: false },
    ];
    expect(experimentsOf('airline', VERSIONS, runs)).toEqual(['v0', 'v3', 'v5', 'v6', 'v10']);
    expect(experimentsOf('telecom', VERSIONS, runs)).toEqual(['v1', 'v2']);
  });

  it('the Agent tab opens the experiment when one is set, else the champion, else the newest', () => {
    const s: Scope = { dataset: 'airline', agent: 'answering', exp: '', task: '' };
    expect(agentVersion(s, VERSIONS, REG)).toBe('v3');
    expect(agentVersion({ ...s, exp: 'v5' }, VERSIONS, REG)).toBe('v5');
    expect(agentVersion({ ...s, dataset: 'telecom' }, VERSIONS, REG)).toBe('v1');
    expect(scopeHref('agent', { ...s, exp: 'v6' }, { versions: VERSIONS, registry: REG })).toBe('/agent/airline/v6');
    expect(scopeHref('agent', { ...s, agent: 'judge' })).toBe('/agent/airline/judge');
  });

  it('every model string here falls in one family', () => {
    expect(['claude-sdk/claude-haiku-4-5', 'claude-sonnet-5', 'opus', 'claude-opus-5-5'].map(modelFamily)).toEqual(['haiku', 'sonnet', 'opus', 'opus']);
  });
});
