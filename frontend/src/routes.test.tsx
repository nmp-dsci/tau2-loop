/**
 * The URL grammar, enforced: every address the viewer ever published still lands,
 * and every address it builds matches the page it names. Drives the real route
 * table in a memory router (loaders and redirects run; nothing renders), plus the
 * id helpers in `lib/url.ts`.
 */
import { createMemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import {
  agentPath,
  domainPath,
  optimisePath,
  parseTaskId,
  parseTrialId,
  patchLens,
  reviewPath,
  runPath,
  search,
  taskId,
  taskPath,
  trialId,
  trialPath,
} from './lib/url';
import { routes } from './routes';

const RUN = '20260915T132148Z_airline_v2_train';
/** One task id per domain. telecom's is the awkward one: brackets, pipes and a colon. */
const TASKS: [string, string][] = [
  ['airline', '0'],
  ['retail', '12'],
  ['banking_knowledge', 'task_001'],
  ['telecom', '[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]'],
];

async function land(url: string): Promise<{ path: string; route: string | undefined }> {
  const router = createMemoryRouter(routes, { initialEntries: [url] });
  await new Promise<void>((resolve) => {
    if (router.state.initialized && router.state.navigation.state === 'idle') return resolve();
    const off = router.subscribe((s) => {
      if (s.initialized && s.navigation.state === 'idle') {
        off();
        resolve();
      }
    });
  });
  const { pathname, search: qs } = router.state.location;
  const route = router.state.matches.at(-1)?.route.id;
  router.dispose();
  return { path: decodeURIComponent(`${pathname}${qs}`), route };
}

describe('addresses from before the grammar still land', () => {
  it.each([
    ['/data', '/domains', 'domains'],
    ['/tasks', '/domains', 'domains'],
    ['/tasks/airline', '/domains/airline', 'domain'],
    ['/tasks/airline/0', '/domains/airline/0', 'task'],
    ['/agents', '/agent', 'agents'],
    ['/agents/airline/v2', '/agent/airline/v2', 'agent'],
    // the three tabs that merged at M3
    ['/architecture', '/agent', 'agents'],
    ['/loop', '/optimise/airline', 'optimise'],
    ['/loop?domain=telecom', '/optimise/telecom', 'optimise'],
    ['/evolution?domain=airline&a=v1&b=v2', '/optimise/airline/v2', 'optimise-round'],
    ['/evolution', '/optimise/airline', 'optimise'],
    ['/compare?a=run-a&b=run-b', '/runs/run-b?vs=run-a', 'run'],
    ['/compare', '/runs', 'runs'],
    ['/optimise', '/optimise/airline', 'optimise'],
    ['/optimise?domain=retail', '/optimise/retail', 'optimise'],
    [`/runs/${RUN}/traces/0.json`, `/runs/${RUN}/0/t1`, 'trial'],
    [`/runs/${RUN}/traces/0_t3.json`, `/runs/${RUN}/0/t3`, 'trial'],
    [`/runs/${RUN}/traces/task_001.json`, `/runs/${RUN}/task_001/t1`, 'trial'],
  ])('%s → %s', async (from, to, route) => {
    const got = await land(from);
    expect(got.path).toBe(to);
    expect(got.route).toBe(route);
  });

  it('an old trace address keeps its lens', async () => {
    const got = await land(`/runs/${RUN}/traces/0_t2.json?node=judge`);
    expect(got.path).toBe(`/runs/${RUN}/0/t2?node=judge`);
  });
});

describe('every address the viewer builds lands on the page it names', () => {
  it('a round opens inside the rounds list, which stays mounted', async () => {
    const router = createMemoryRouter(routes, { initialEntries: [optimisePath('airline', 'v2')] });
    await new Promise<void>((resolve) => {
      if (router.state.initialized) return resolve();
      const off = router.subscribe((s) => {
        if (s.initialized) {
          off();
          resolve();
        }
      });
    });
    // both the list route and the round route are matched: the round renders into the list's Outlet
    expect(router.state.matches.map((m) => m.route.id)).toEqual([
      'shell',
      'optimise',
      'optimise-round',
    ]);
    router.dispose();
  });

  it('the overview is the root', async () => {
    expect(await land('/')).toEqual({ path: '/', route: 'overview' });
  });

  it.each(TASKS)('a %s task address lands on the task page', async (domain, id) => {
    expect((await land(domainPath(domain))).route).toBe('domain');
    const got = await land(taskPath(taskId(domain, id)));
    expect(got.route).toBe('task');
    expect(got.path).toBe(`/domains/${domain}/${id}`);
  });

  it.each(TASKS)('a %s review address encodes its task id exactly once', async (_domain, id) => {
    const got = await land(reviewPath(RUN, `${id}/t1`));
    expect(got.route).toBe('review-one');
    // decoded once, it is the id again: a double-encoded `[` would read %5B here
    expect(got.path).toBe(`/review/${RUN}/${id}/t1`);
  });

  it.each(TASKS)('a %s trial address lands on the trace page', async (_domain, id) => {
    const got = await land(trialPath(RUN, trialId({ task_id: id, trial: 2 })));
    expect(got.route).toBe('trial');
    expect(got.path).toBe(`/runs/${RUN}/${id}/t2`);
  });

  it.each([
    [runPath(RUN), 'run'],
    [runPath(RUN, { vs: 'other-run' }), 'run'],
    ['/runs', 'runs'],
    [agentPath('airline', 'v2'), 'agent'],
    ['/agent', 'agents'],
    [agentPath('airline', 'v2', { node: 'judge' }), 'agent'],
    ['/rubric', 'rubric'],
    ['/review', 'review'],
    [`/review/${RUN}/0/t1`, 'review-one'],
    [reviewPath(RUN, '0/t2'), 'review-one'],
    [reviewPath(undefined, undefined, { run: RUN }), 'review'],
    [optimisePath('telecom'), 'optimise'],
    [optimisePath('airline', 'v2'), 'optimise-round'],
    [optimisePath('airline', 'v2', { step: 'outcome' }), 'optimise-round'],
  ])('%s → %s', async (url, route) => {
    expect((await land(url)).route).toBe(route);
  });
});

describe('the id helpers', () => {
  it('a task id is its domain and the rest, split once', () => {
    for (const [domain, id] of TASKS) {
      expect(parseTaskId(taskId(domain, id))).toEqual({ domain, task: id });
    }
  });

  it('a trial id round-trips, telecom pipes and all', () => {
    for (const [, id] of TASKS) {
      expect(parseTrialId(trialId({ task_id: id, trial: 4 }))).toEqual({ task: id, trial: 4 });
    }
  });

  it('is not fooled by a task id that is not one', () => {
    expect(parseTaskId('airline')).toBeNull();
    expect(parseTrialId('0')).toBeNull();
  });

  it('a lens skips empty values and keeps slashes readable', () => {
    expect(search({ a: '1', b: '', c: null, d: 'x/y' })).toBe('?a=1&d=x/y');
    expect(search({})).toBe('');
  });

  it('patching a lens removes a key when the value is empty', () => {
    const sp = new URLSearchParams('a=1&b=2');
    expect(patchLens(sp, { b: '' })).toBe('?a=1');
    expect(patchLens(sp, { c: '3' })).toBe('?a=1&b=2&c=3');
  });
});
