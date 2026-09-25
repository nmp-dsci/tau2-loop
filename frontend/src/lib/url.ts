/**
 * One address per thing. Every viewer URL is built here, from three rules
 * (the same three DataAgentBench's `lib/url.ts` states, applied to τ²'s ids):
 *
 * 1. One id, the same string everywhere. Domain `airline`, task `airline/0`
 *    (as every table prints it), trial `airline/0/t1` inside a run, run
 *    `20260915T132148Z_airline_v2_train`, agent version `airline/v2`. In a path
 *    the id's slashes are path separators, so an address can be chopped back one
 *    level at a time. The trace file name (`0_t2.json`, written by
 *    `eval/runner.py`) stays on disk and never reaches a URL.
 * 2. The path names the subject; the query string holds the lens. If dropping a
 *    value leaves nothing to show, it is the subject (a domain, a task, a run, a
 *    trial, an agent version) and goes in the path. If dropping it falls back to
 *    a default (a selected file, a compared run, a filter) it is a lens.
 * 3. Picking from a list opens the detail in place: a nested route renders into
 *    the list page's <Outlet>, so the address changes and the list stays.
 *
 * τ² task ids are not all tidy: telecom's carry `[`, `]`, `|` and `:`
 * (`[mobile_data_issue]airplane_mode_on|…[PERSONA:Hard]`). None contains a
 * slash, so a task id is exactly two segments and `enc` below keeps the
 * readable characters readable.
 */

import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';

export type Lens = Record<string, string | null | undefined>;

// ── ids ───────────────────────────────────────────────────────────────────
/** `airline` + `0` → `airline/0`. The id every table, API and URL prints. */
export function taskId(domain: string, id: string): string {
  return `${domain}/${id}`;
}

/** `airline/0` → its parts, or null when it is not a task id. */
export function parseTaskId(id: string): { domain: string; task: string } | null {
  const i = id.indexOf('/');
  return i > 0 ? { domain: id.slice(0, i), task: id.slice(i + 1) } : null;
}

/** One conversation of one run: task id + trial, `0/t1`. The run supplies the domain. */
export function trialId(r: { task_id: string; trial: number }): string {
  return `${r.task_id}/t${r.trial}`;
}

/** `0/t1` → its parts, or null when it is not a trial id. */
export function parseTrialId(id: string): { task: string; trial: number } | null {
  const m = /^(.+)\/t(\d+)$/.exec(id);
  return m ? { task: m[1], trial: Number(m[2]) } : null;
}

// ── paths ─────────────────────────────────────────────────────────────────
export const domainsPath = (lens?: Lens): string => `/domains${search(lens)}`;
export const domainPath = (domain: string, lens?: Lens): string =>
  `/domains/${enc(domain)}${search(lens)}`;
/** A task lives under its domain: `/domains/airline/0`. */
export const taskPath = (id: string, lens?: Lens): string => `/domains/${encPath(id)}${search(lens)}`;
export const runsPath = (lens?: Lens): string => `/runs${search(lens)}`;
export const runPath = (runId: string, lens?: Lens): string => `/runs/${enc(runId)}${search(lens)}`;
/** One conversation of one run: `/runs/<run>/0/t1`. */
export const trialPath = (runId: string, tid: string, lens?: Lens): string =>
  `/runs/${enc(runId)}/${encPath(tid)}${search(lens)}`;
export const agentsPath = (lens?: Lens): string => `/agent${search(lens)}`;
/** A round is named for the version it wrote: `/optimise/airline/v2`. */
export const optimisePath = (domain?: string, version?: string, lens?: Lens): string =>
  (domain ? `/optimise/${enc(domain)}${version ? `/${enc(version)}` : ''}` : '/optimise') + search(lens);
export const rubricPath = (lens?: Lens): string => `/rubric${search(lens)}`;
/** The review list, or one conversation's review: `/review/<run>/<task>/t1`.
 *  Takes the ids, never an already-built path: encoding an encoded segment again is
 *  how `%5B` becomes `%255B`. */
export const reviewPath = (runId?: string, tid?: string, lens?: Lens): string =>
  (runId && tid ? `/review/${enc(runId)}/${encPath(tid)}` : '/review') + search(lens);
export const agentPath = (domain: string, version: string, lens?: Lens): string =>
  `/agent/${enc(domain)}/${enc(version)}${search(lens)}`;

// ── encoding ──────────────────────────────────────────────────────────────
/** Escape one path segment, leaving `:` and `,` readable (both legal in a path
 *  per RFC 3986 §3.3). `[`, `]` and `|` are not legal unencoded and stay escaped,
 *  which is what telecom's task ids need. */
function enc(v: string): string {
  return encodeURIComponent(v).replace(/%3A/gi, ':').replace(/%2C/gi, ',');
}

/** An id whose slashes are separators: each segment escaped, the slashes kept. */
function encPath(id: string): string {
  return id.split('/').map(enc).join('/');
}

/** Escape a query-string value, but leave `/`, `:` and `,` readable (all legal in a
 *  query per RFC 3986 §3.4), so a lens reads `?vs=20260915T132148Z_airline_v2_train`. */
function encQuery(v: string): string {
  return encodeURIComponent(v).replace(/%2F/gi, '/').replace(/%3A/gi, ':').replace(/%2C/gi, ',');
}

// ── the lens ──────────────────────────────────────────────────────────────
/** `?a=1&b=x/y`, skipping empty values; '' when nothing is set. Keys keep their given order. */
export function search(lens?: Lens): string {
  const parts = Object.entries(lens ?? {})
    .filter((e): e is [string, string] => e[1] != null && e[1] !== '')
    .map(([k, v]) => `${encQuery(k)}=${encQuery(v)}`);
  return parts.length ? `?${parts.join('&')}` : '';
}

/** The current lens with `patch` applied (null or '' removes a key), as a `?…` string. */
export function patchLens(current: URLSearchParams, patch: Lens): string {
  const next: Lens = Object.fromEntries(current.entries());
  for (const [k, v] of Object.entries(patch)) next[k] = v;
  return search(next);
}

/** The page's lens and a setter that rewrites it in place (replace, not push), readable. */
export function useLens(): [URLSearchParams, (patch: Lens) => void] {
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const { pathname, hash } = useLocation();
  const set = (patch: Lens) => nav({ pathname, search: patchLens(sp, patch), hash }, { replace: true });
  return [sp, set];
}
