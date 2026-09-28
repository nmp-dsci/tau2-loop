import { useEffect, useState } from 'react';

export const DOMAINS = ['airline', 'retail', 'telecom', 'banking_knowledge'] as const;
export type Domain = (typeof DOMAINS)[number];

export type Health = { status: string; mode: 'demo' | 'live'; champions: Record<string, string | null>; code_sha: string; domains: string[]; mlflow_url?: string; mlflow_embeddable?: boolean; writable?: boolean; playground?: boolean };
export type Summary = {
  n: number;
  n_scored: number;
  passed: number;
  pass_rate: number | null;
  n_tasks: number;
  trials: number;
  pass_hat_k: Record<string, number>;
  failed_ids: string[];
  errored_ids: string[];
  by_termination: Record<string, number>;
  mean_agent_turns: number;
  mean_tool_calls: number;
  partial_action_mean: number | null;
  duration_ms: number;
  cost_usd_est: number;
  /** per reward check; absent on a run scored before the field existed (`/api/runs` fills it) */
  checks?: Checks;
};
/** One reward check over a run: conversations carrying it that met every item (`passed / n`),
 * the items themselves, and how many of those conversations the domain's score multiplies it into. */
export type Check = { passed: number; n: number; items_met: number; items: number; scored: number };
export type Checks = Partial<Record<'db' | 'actions' | 'communicate' | 'nl' | 'env', Check>>;
export type RunMeta = {
  run_id: string;
  domain: string;
  agent: string;
  fingerprint: string;
  model: string;
  user_model: string;
  judge_model: string;
  split: string;
  n_tasks: number;
  trials: number;
  concurrency: number;
  seed: number;
  started_at: string;
  finished_at: string | null;
  code_sha: string;
  tau2_sha: string;
  summary: Summary | null;
  mlflow_run_id: string | null;
  note: string;
  task_ids: string[];
  tool_mode: string;
  sampling: string;
  dry_run: boolean;
  mlflow_url?: string | null;
  /** mean tokens per conversation, all roles and the agent's share; `/api/runs` only */
  tokens_per_conversation?: { all: number; agent: number } | null;
  /** since split v2; absent (or null) on older run.json files */
  agent_effort?: string | null;
  user_effort?: string | null;
  /** 1 = the 20 / 20 cut, 2 = half of each base set; `/api/runs` infers it for older runs */
  split_version?: number | null;
  /** `in-process`, or `service:<host>` when the agent's calls went over HTTP */
  agent_route?: string;
  /** `/api/runs` only: the summary's checks, or aggregated from the rows for older runs */
  checks?: Checks;
};
export type TaskResult = {
  task_id: string;
  trial: number;
  reward: number;
  correct: boolean | null;
  reward_basis: string[];
  db_check: boolean | null;
  env_assertions: string | null;
  action_checks: string | null;
  communicate_checks: string | null;
  nl_assertions: string | null;
  partial_action_reward: number | null;
  termination_reason: string;
  n_messages: number;
  n_agent_turns: number;
  n_tool_calls: number;
  n_tool_errors: number;
  agent_input_tokens: number;
  agent_output_tokens: number;
  user_input_tokens: number;
  user_output_tokens: number;
  duration_ms: number;
  cost_usd_est: number | null;
  error: string | null;
  purpose: string;
  trace: string;
};
export type Diagnosis = { task_id: string; symptom: string; root_cause: string; surface: string; change: string; verified_in_session: boolean; verification?: string };
/** Champion vs challenger on the test split: reported beside the verdict, never used by it. */
export type TestCompare = { champion_run: string; challenger_run?: string; passes?: string; pass_1?: string; fixed?: string[]; broken?: string[]; p_value?: number; reason?: string; error?: string };
export type Outcome = { verdict: string; reason?: string; rule?: string | null; p_value?: number; passes?: string; pass_1?: string; fixed?: string[]; broken?: string[]; still_failed?: string[]; challenger_run?: string; test_run?: string; test_passes?: string; test_compare?: TestCompare };
export type LedgerEntry = {
  cycle: number;
  domain: string;
  champion: string;
  champion_run?: string;
  challenger: string | null;
  optimiser_model?: string;
  failed: string[];
  diagnoses?: Diagnosis[];
  prompt_diff_summary?: string;
  helper_diff_summary?: string;
  expected_to_fix?: string[];
  risks?: string[];
  optimiser?: { turns: number; duration_ms: number; cost_usd_est: number | null; error: string | null };
  tokens?: Record<string, number>;
  outcome?: Outcome;
  at?: string;
};
export type RegistryEntry = { agent: string; fingerprint: string; run_id: string; model: string; split: string; trials: number; passed: number | null; n_scored: number | null; at: string };
export type Registry = { domain: string; champion: RegistryEntry | null; challenger: RegistryEntry | null; history: (RegistryEntry & { event: string })[] };
export type Registries = Record<string, Registry>;
export type AgentInfo = { domain: string; name: string; ref: string; fingerprint: string; config: Record<string, unknown>; has_helper: boolean; helper_functions: string[]; diagnosis: Record<string, unknown> | null; runs: string[] };
export type DomainSummary = {
  domain: string;
  base_n: number | null;
  train: number;
  test: number;
  reserve_n: number | null;
  seed: number | null;
  split_version?: number;
  policy_words: number | null;
  n_tools: number;
  reward_bases: Record<string, number>;
  champion: RegistryEntry | null;
  challenger: RegistryEntry | null;
  versions: string[];
  runs: number;
  cycles: number;
};
export type TaskRow = { id: string; split: string; purpose: string | null; relevant_policies: string | null; reason_for_call: string | null; n_actions: number; n_communicate: number; n_nl_assertions: number; n_env_assertions: number; reward_basis: string[] | null };
export type DomainDetail = { domain: string; split: { seed: number; version?: number; base_n: number; method: string; train: string[]; test: string[]; reserve_n: number }; policy: string; policy_words: number; tools: { name: string; description: string | null }[]; tasks: TaskRow[] };
export type Event = { type: string; [k: string]: unknown };

// ── one conversation, whole (the Agent tab) ──────────────────────────────
/** A domain tool with tau2's own type; `mutates` is whether a state rebuild re-runs it. */
export type ToolSpec = { name: string; description: string | null; type: 'read' | 'write' | 'think' | 'generic'; mutates: boolean };
export type TraceCall = { id: string | null; name: string; arguments: Record<string, unknown>; requestor: string };
export type TraceMessage = {
  i: number;
  role: 'assistant' | 'user' | 'tool' | 'system';
  content: string | null;
  tool_calls: TraceCall[];
  /** a tool result's id is the id of the call it answers */
  id: string | null;
  requestor: string | null;
  turn_idx: number | null;
  usage: { prompt_tokens: number | null; completion_tokens: number | null } | null;
  seconds: number | null;
  error: boolean;
};
export type ExpectedAction = { action_id: string; name: string; arguments: Record<string, unknown>; requestor?: string };
export type RewardInfo = {
  reward: number;
  reward_basis?: string[] | null;
  reward_breakdown?: Record<string, number> | null;
  db_check?: { db_match: boolean; db_reward: number } | null;
  action_checks?: { action: ExpectedAction; action_match: boolean; action_reward?: number; tool_type?: string | null }[] | null;
  communicate_checks?: { info: string; met: boolean; justification?: string }[] | null;
  nl_assertions?: { nl_assertion: string; met: boolean; justification: string }[] | null;
  env_assertions?: { env_assertion: { func_name?: string; arguments?: unknown }; met: boolean; reward?: number }[] | null;
  info?: Record<string, unknown> | null;
};
export type Scenario = { reason_for_call?: string | null; known_info?: string | null; unknown_info?: string | null; task_instructions?: string | null };
export type TaskSpec = {
  id: string;
  description?: { purpose?: string | null; relevant_policies?: string | null } | null;
  user_scenario?: { instructions?: Scenario | string | null } | null;
  evaluation_criteria?: {
    actions?: ExpectedAction[] | null;
    nl_assertions?: string[] | null;
    communicate_info?: string[] | null;
    env_assertions?: unknown[] | null;
    reward_basis?: string[] | null;
  } | null;
  user_tools?: string[] | null;
};
export type TrialPayload = {
  task_id: string;
  trial: number | null;
  termination_reason: string;
  duration: number;
  reward_info: RewardInfo | null;
  events: Event[];
  policy_words: number;
  result: TaskResult;
  domain: string;
  messages: TraceMessage[];
  task: TaskSpec | null;
  tools: ToolSpec[];
  user_tools: ToolSpec[];
};
/** The agent a run ran with: its snapshot, and the prompt composed exactly as the agent did. */
export type RunAgent = {
  run_id: string;
  domain: string;
  agent: string;
  fingerprint: string;
  config: Record<string, unknown>;
  files: Record<string, string>;
  hooks: Record<string, boolean>;
  prompt: { text: string; system_md_chars: number; policy: string; policy_words: number; extra_context: string | null; slotted: boolean };
};
export type DiffRecord = { record: string; fields: { field: string; before: string | null; after: string | null }[] };
export type PlaygroundResult = {
  name: string;
  arguments: Record<string, unknown>;
  at: number;
  after_calls: number;
  requestor: string;
  content: string;
  error: boolean;
  wrote: boolean;
  diff: DiffRecord[];
  replayed_writes: number;
  same_as_recorded: boolean | null;
};

export async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return (await r.json()) as T;
}

/** `nonce` re-fetches the same URL: bump it after a write, so a page that changed
 *  something server-side reads the new state without changing its address. */
export function useGet<T>(url: string | null, nonce = 0): { data: T | null; error: string | null; loading: boolean } {
  const [state, set] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: !!url });
  useEffect(() => {
    if (!url) return;
    let alive = true;
    set({ data: null, error: null, loading: true });
    get<T>(url)
      .then((d) => alive && set({ data: d, error: null, loading: false }))
      .catch((e: Error) => alive && set({ data: null, error: e.message, loading: false }));
    return () => {
      alive = false;
    };
  }, [url, nonce]);
  return state;
}

export async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail ?? `${r.status} ${url}`);
  return j as T;
}

export function fmtPct(x: number | null | undefined, digits = 0): string {
  return x == null ? '—' : `${(x * 100).toFixed(digits)}%`;
}
export function fmtS(ms: number | null | undefined): string {
  return ms == null ? '—' : ms >= 60000 ? `${(ms / 60000).toFixed(1)}m` : `${(ms / 1000).toFixed(0)}s`;
}
export function fmtK(n: number | null | undefined): string {
  if (n == null) return '—';
  return n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n);
}
export function shortRun(id: string): string {
  return id.replace(/^\d{8}T\d{6}Z_/, '');
}
export function shortModel(m: string): string {
  return m.replace(/^claude-sdk\//, '').replace(/^claude-/, '');
}
export function when(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toISOString().slice(0, 16).replace('T', ' ') + 'Z';
}
export function shortTask(id: string, n = 36): string {
  return id.length > n ? id.slice(0, n - 1) + '…' : id;
}
export function domainLabel(d: string): string {
  return d === 'banking_knowledge' ? 'banking' : d;
}
