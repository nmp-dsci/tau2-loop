import { type RouteObject, redirect } from 'react-router-dom';
import { Shell } from './Shell';
import { Overview } from './pages/Overview';
import { Data } from './pages/Data';
import { Tasks } from './pages/Tasks';
import { Architecture } from './pages/Architecture';
import { Agents } from './pages/Agents';
import { Runs } from './pages/Runs';
import { Run } from './pages/Run';
import { Trace } from './pages/Trace';
import { Compare } from './pages/Compare';
import { Loop } from './pages/Loop';
import { Evolution } from './pages/Evolution';
import { search } from './lib/url';

/**
 * Every viewer address, as data (`routes.test.tsx` drives each one in a memory
 * router). The grammar is in `lib/url.ts`. Addresses from before the grammar are
 * redirect loaders, never 404s: a bookmark, a review note or a README link from
 * before keeps landing.
 */

/** `/runs/<id>/traces/<file>` → the trial the file holds. `eval/runner.py` names a
 *  trace `<slug(task_id)>.json`, or `<slug>_t<trial>.json` when a run has trials;
 *  the slug is lossy, so the redirect keeps the stem as the task and trusts the
 *  trial page to resolve it against the run. */
function trialFromTraceFile(name: string): { task: string; trial: number } {
  const stem = name.replace(/\.json$/i, '');
  const m = /^(.+)_t(\d+)$/.exec(stem);
  return m ? { task: m[1], trial: Number(m[2]) } : { task: stem, trial: 1 };
}

export const routes: RouteObject[] = [
  {
    id: 'shell',
    element: <Shell />,
    hydrateFallbackElement: <></>,
    children: [
      { id: 'overview', path: '/', element: <Overview /> },

      // the benchmark: a domain, then one task inside it
      { id: 'domains', path: '/domains', element: <Data /> },
      { id: 'domain', path: '/domains/:domain', element: <Tasks /> },
      { id: 'task', path: '/domains/:domain/:taskId', element: <Tasks /> },

      // what we ran
      { id: 'runs', path: '/runs', element: <Runs /> },
      { id: 'run', path: '/runs/:runId', element: <Run /> },
      { id: 'trial', path: '/runs/:runId/:taskId/:trial', element: <Trace /> },

      // the agent
      { id: 'agents', path: '/agent', element: <Agents /> },
      { id: 'agent', path: '/agent/:domain/:name', element: <Agents /> },
      { id: 'architecture', path: '/architecture', element: <Architecture /> },

      // the loop
      { id: 'compare', path: '/compare', element: <Compare /> },
      { id: 'loop', path: '/loop', element: <Loop /> },
      { id: 'evolution', path: '/evolution', element: <Evolution /> },

      // ── addresses from before the grammar ──
      { id: 'old-data', path: '/data', loader: () => redirect('/domains') },
      { id: 'old-tasks', path: '/tasks', loader: () => redirect('/domains') },
      {
        id: 'old-tasks-domain',
        path: '/tasks/:domain',
        loader: ({ params }) => redirect(`/domains/${encodeURIComponent(params.domain!)}`),
      },
      {
        id: 'old-task',
        path: '/tasks/:domain/:taskId',
        loader: ({ params }) =>
          redirect(`/domains/${encodeURIComponent(params.domain!)}/${encodeURIComponent(params.taskId!)}`),
      },
      { id: 'old-agents', path: '/agents', loader: () => redirect('/agent') },
      {
        id: 'old-agent',
        path: '/agents/:domain/:name',
        loader: ({ params }) =>
          redirect(`/agent/${encodeURIComponent(params.domain!)}/${encodeURIComponent(params.name!)}`),
      },
      {
        id: 'old-trace',
        path: '/runs/:runId/traces/:name',
        loader: ({ params, request }) => {
          const { task, trial } = trialFromTraceFile(params.name!);
          const q = Object.fromEntries(new URL(request.url).searchParams.entries());
          return redirect(
            `/runs/${encodeURIComponent(params.runId!)}/${encodeURIComponent(task)}/t${trial}${search(q)}`,
          );
        },
      },
    ],
  },
];
