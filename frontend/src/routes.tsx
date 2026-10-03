import { type LoaderFunctionArgs, type RouteObject, redirect } from 'react-router-dom';
import { Shell } from './Shell';
import { Overview } from './pages/Overview';
import { Domain, Domains } from './pages/Domains';
import { JudgeEvals } from './pages/JudgeEvals';
import { Rubric } from './pages/Rubric';
import { Agent } from './pages/Agent';
import { Runs } from './pages/Runs';
import { Run } from './pages/Run';
import { Trace } from './pages/Trace';
import { Optimise, OptimiseRound } from './pages/Optimise';
import { JudgeLoop } from './pages/JudgeLoop';
import { JudgeAgent } from './pages/JudgeAgent';
import { Review } from './pages/Review';
import { GoldReview } from './pages/GoldReview';
import { Leaderboard } from './pages/Leaderboard';
import { search } from './lib/url';

/**
 * Every viewer address, as data (`routes.test.tsx` drives each one in a memory
 * router). The grammar is in `lib/url.ts`. Addresses from before the grammar are
 * redirect loaders, never 404s: a bookmark, a review note or a README link from
 * before keeps landing.
 */

/** `/runs/<id>/traces/<file>` → the trial the file holds. `eval/runner.py` names a
 *  trace `<slug(task_id)>.json`, or `<slug>_t<trial>.json` when a run has trials;
 *  the slug is lossy, so the redirect keeps the stem as the task and the API
 *  resolves it against the run. */
function trialFromTraceFile(name: string): { task: string; trial: number } {
  const stem = name.replace(/\.json$/i, '');
  const m = /^(.+)_t(\d+)$/.exec(stem);
  return m ? { task: m[1], trial: Number(m[2]) } : { task: stem, trial: 1 };
}

/** The old Gate tab was `?a=<champion>&b=<challenger>`; the gate now lives at the
 *  challenger's own address, with the champion as its lens. */
function gateAddress(url: URL): string {
  const a = url.searchParams.get('a') ?? '';
  const b = url.searchParams.get('b') ?? '';
  if (!b) return '/runs';
  return `/runs/${encodeURIComponent(b)}${search({ vs: a })}`;
}

export const routes: RouteObject[] = [
  {
    id: 'shell',
    element: <Shell />,
    hydrateFallbackElement: <></>,
    children: [
      { id: 'overview', path: '/', element: <Overview /> },

      // the benchmark: a domain, then one task inside it
      // the evals (renamed from Domains & tasks): a dataset's eval set, for either agent
      { id: 'evals', path: '/evals', element: <Domains /> },
      { id: 'evals-domain', path: '/evals/:domain', element: <Domain /> },
      // the LLM judge's eval set (s11); `judge` outranks a task id
      { id: 'evals-judge', path: '/evals/:domain/judge', element: <JudgeEvals /> },
      { id: 'evals-judge-conv', path: '/evals/:domain/judge/:runId/:taskId/:trial', element: <JudgeEvals /> },
      { id: 'evals-task', path: '/evals/:domain/:taskId', element: <Domain /> },

      // how a conversation is judged
      { id: 'rubric', path: '/rubric', element: <Rubric /> },

      // who else has tried
      { id: 'leaderboard', path: '/leaderboard', element: <Leaderboard /> },

      // what we ran — the gate is a lens on a run, not a tab
      { id: 'runs', path: '/runs', element: <Runs /> },
      { id: 'run', path: '/runs/:runId', element: <Run /> },
      { id: 'trial', path: '/runs/:runId/:taskId/:trial', element: <Trace /> },

      // how it improves: a round opens inside the rounds list (grammar rule 3)
      {
        id: 'optimise-bare',
        path: '/optimise',
        loader: ({ request }: LoaderFunctionArgs) => {
          const d = new URL(request.url).searchParams.get('domain');
          return redirect(`/optimise/${encodeURIComponent(d ?? 'airline')}`);
        },
      },
      // the tool judge's own loop beside the agent's (s11); `judge` outranks a `:version`
      { id: 'optimise-judge', path: '/optimise/:domain/judge', element: <JudgeLoop /> },
      {
        id: 'optimise',
        path: '/optimise/:domain',
        element: <Optimise />,
        children: [{ id: 'optimise-round', path: ':version', element: <OptimiseRound /> }],
      },

      // what a person thought of what the judge scored — the one write path
      { id: 'review', path: '/review', element: <Review /> },
      { id: 'review-one', path: '/review/:runId/:taskId/:trial', element: <Review /> },
      // the tool judge's answer key (s11 J1): a person checks what structure could not pin
      { id: 'review-golden', path: '/review/golden/:domain', element: <GoldReview /> },
      { id: 'review-golden-one', path: '/review/golden/:domain/:runId/:taskId/:trial/:msg', element: <GoldReview /> },

      // what the agent is
      { id: 'agents', path: '/agent', element: <Agent /> },
      // the LLM judge as an agent of its own (s11); `judge` outranks a version name
      { id: 'agent-judge', path: '/agent/:domain/judge', element: <JudgeAgent /> },
      { id: 'agent', path: '/agent/:domain/:name', element: <Agent /> },

      // ── addresses from before the grammar ──
      { id: 'old-data', path: '/data', loader: () => redirect('/evals') },
      { id: 'old-tasks', path: '/tasks', loader: () => redirect('/evals') },
      {
        id: 'old-tasks-domain',
        path: '/tasks/:domain',
        loader: ({ params }) => redirect(`/evals/${encodeURIComponent(params.domain!)}`),
      },
      {
        id: 'old-task',
        path: '/tasks/:domain/:taskId',
        loader: ({ params }) =>
          redirect(
            `/evals/${encodeURIComponent(params.domain!)}/${encodeURIComponent(params.taskId!)}`,
          ),
      },
      // Domains & tasks became Evals (s11 review): its addresses keep landing, lens and all
      {
        id: 'old-domains',
        path: '/domains',
        loader: ({ request }: LoaderFunctionArgs) => redirect(`/evals${new URL(request.url).search}`),
      },
      {
        id: 'old-domains-domain',
        path: '/domains/:domain',
        loader: ({ params, request }) =>
          redirect(`/evals/${encodeURIComponent(params.domain!)}${new URL(request.url).search}`),
      },
      {
        id: 'old-domains-task',
        path: '/domains/:domain/:taskId',
        loader: ({ params, request }) =>
          redirect(
            `/evals/${encodeURIComponent(params.domain!)}/${encodeURIComponent(params.taskId!)}${new URL(request.url).search}`,
          ),
      },
      { id: 'old-agents', path: '/agents', loader: () => redirect('/agent') },
      {
        id: 'old-agent',
        path: '/agents/:domain/:name',
        loader: ({ params }) =>
          redirect(`/agent/${encodeURIComponent(params.domain!)}/${encodeURIComponent(params.name!)}`),
      },
      // Architecture became the figure at the top of the Agent tab
      { id: 'old-architecture', path: '/architecture', loader: () => redirect('/agent') },
      // Loop and Evolution became one round
      {
        id: 'old-loop',
        path: '/loop',
        loader: ({ request }: LoaderFunctionArgs) => {
          const d = new URL(request.url).searchParams.get('domain') ?? 'airline';
          return redirect(`/optimise/${encodeURIComponent(d)}`);
        },
      },
      {
        id: 'old-evolution',
        path: '/evolution',
        loader: ({ request }: LoaderFunctionArgs) => {
          const sp = new URL(request.url).searchParams;
          const d = sp.get('domain') ?? 'airline';
          const b = sp.get('b');
          return redirect(
            `/optimise/${encodeURIComponent(d)}${b ? `/${encodeURIComponent(b)}` : ''}`,
          );
        },
      },
      // the Gate tab became a lens on the challenger's run
      {
        id: 'old-compare',
        path: '/compare',
        loader: ({ request }: LoaderFunctionArgs) => redirect(gateAddress(new URL(request.url))),
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
