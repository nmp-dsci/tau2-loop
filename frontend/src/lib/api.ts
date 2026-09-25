import { useEffect, useState } from 'react';

export const DOMAINS = ['airline', 'retail', 'telecom', 'banking_knowledge'] as const;
export type Domain = (typeof DOMAINS)[number];

export type Health = { status: string; mode: 'demo' | 'live'; champions: Record<string, string | null>; code_sha: string; domains: string[]; mlflow_url?: string; mlflow_embeddable?: boolean };
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
};
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
export type Outcome = { verdict: string; reason?: string; p_value?: number; passes?: string; fixed?: string[]; broken?: string[]; still_failed?: string[]; challenger_run?: string; test_run?: string; test_passes?: string };
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
export type DomainDetail = { domain: string; split: { seed: number; size: number; base_n: number; method: string; train: string[]; test: string[]; reserve_n: number }; policy: string; policy_words: number; tools: { name: string; description: string | null }[]; tasks: TaskRow[] };
export type Event = { type: string; [k: string]: unknown };

export async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return (await r.json()) as T;
}

export function useGet<T>(url: string | null): { data: T | null; error: string | null; loading: boolean } {
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
  }, [url]);
  return state;
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
