// Thin typed API client. The API key (if the backend requires one) is entered by the
// user and kept in sessionStorage; AI provider keys never reach the browser.

const BASE = (import.meta as any).env?.VITE_API_BASE ?? "";

function apiKey(): string | null {
  try { return sessionStorage.getItem("psx_api_key"); } catch { return null; }
}
export function setApiKey(k: string) {
  try { sessionStorage.setItem("psx_api_key", k); } catch { /* storage unavailable */ }
}

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j); } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const k = apiKey();
  return k ? { ...extra, "X-API-Key": k } : extra;
}

export async function get<T>(path: string): Promise<T> {
  return handle<T>(await fetch(BASE + path, { headers: headers() }));
}
export async function post<T>(path: string, body: unknown): Promise<T> {
  return handle<T>(await fetch(BASE + path, { method: "POST", headers: headers({ "Content-Type": "application/json" }), body: JSON.stringify(body) }));
}
export async function postForm<T>(path: string, form: FormData): Promise<T> {
  return handle<T>(await fetch(BASE + path, { method: "POST", headers: headers(), body: form }));
}
export async function del<T>(path: string): Promise<T> {
  return handle<T>(await fetch(BASE + path, { method: "DELETE", headers: headers() }));
}
export const reportUrl = (sym: string, tf: string) => `${BASE}/api/report/${encodeURIComponent(sym)}?timeframe=${tf}`;

export const fmt = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
export const stageLabel = (s: string | null | undefined) => (s ? s.replace(/_/g, " ") : "—");
