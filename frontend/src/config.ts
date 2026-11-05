export const APP_NAME = "سعّر";
export const APP_BRAND_MARK = "/ui/saer-header.png";
export const APP_LOGO = "/ui/saer-logo.png";

// يمكن ضبط رابط API وقت البناء أو تمريره عبر ?api= دون إعادة بناء الواجهة.
export const API_BASE_URL = import.meta.env.VITE_GRADIO_SPACE_URL || import.meta.env.VITE_API_BASE_URL || "";
// مصدر JSON ثابت اختياري. يدعم ملف JSON عادي أو Gist API عام.
export const API_CONFIG_URL = import.meta.env.VITE_API_CONFIG_URL || "";

const API_SESSION_KEY = "saer.gradioApiBaseUrl";
const API_CONFIG_SESSION_KEY = "saer.gradioApiConfigUrl";
const CONFIG_CACHE_MS = 15_000;
let configCacheUrl = "";
let configCacheAt = 0;
let configCacheValue = "";
let configFetch: Promise<string> | undefined;

function normalizeApiBaseUrl(value: string): string | null {
  try {
    const url = new URL(value.trim());
    if (!["https:", "http:"].includes(url.protocol) || url.username || url.password) return null;
    if (url.protocol !== "https:" && !["localhost", "127.0.0.1"].includes(url.hostname)) return null;
    url.search = "";
    url.hash = "";
    return url.toString().replace(/\/$/, "");
  } catch {
    return null;
  }
}

function normalizeConfigUrl(value: string): string | null {
  try {
    const url = new URL(value.trim());
    if (!["https:", "http:"].includes(url.protocol) || url.username || url.password) return null;
    if (url.protocol !== "https:" && !["localhost", "127.0.0.1"].includes(url.hostname)) return null;
    return url.toString();
  } catch {
    return null;
  }
}

function queryApiUrl(): string | null {
  if (typeof window === "undefined") return null;
  const value = new URLSearchParams(window.location.search).get("api");
  return value ? normalizeApiBaseUrl(value) : null;
}

export function getApiBaseUrl(): string {
  if (typeof window === "undefined") return normalizeApiBaseUrl(API_BASE_URL) || "";
  const fromQuery = queryApiUrl();
  if (fromQuery) {
    window.sessionStorage.setItem(API_SESSION_KEY, fromQuery);
    return fromQuery;
  }
  return normalizeApiBaseUrl(window.sessionStorage.getItem(API_SESSION_KEY) || "")
    || normalizeApiBaseUrl(API_BASE_URL)
    || "";
}

export function getApiConfigUrl(): string {
  if (typeof window === "undefined") return normalizeConfigUrl(API_CONFIG_URL) || "";
  const queryValue = new URLSearchParams(window.location.search).get("apiConfig");
  const queryUrl = queryValue ? normalizeConfigUrl(queryValue) : null;
  if (queryUrl) {
    window.sessionStorage.setItem(API_CONFIG_SESSION_KEY, queryUrl);
    return queryUrl;
  }
  return normalizeConfigUrl(window.sessionStorage.getItem(API_CONFIG_SESSION_KEY) || "")
    || normalizeConfigUrl(API_CONFIG_URL)
    || "";
}

function apiUrlFromPayload(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  for (const key of ["api_base_url", "api_url", "api"]) {
    if (typeof record[key] === "string") {
      const value = normalizeApiBaseUrl(record[key] as string);
      if (value) return value;
    }
  }

  // GitHub's public Gist API wraps the small config file under files.<name>.content.
  const files = record.files;
  if (files && typeof files === "object") {
    for (const file of Object.values(files as Record<string, unknown>)) {
      if (!file || typeof file !== "object") continue;
      const content = (file as Record<string, unknown>).content;
      if (typeof content !== "string") continue;
      try {
        const nested = JSON.parse(content) as unknown;
        const value = apiUrlFromPayload(nested);
        if (value) return value;
      } catch {
        // Ignore unrelated Gist files and continue searching.
      }
    }
  }
  return null;
}

async function fetchConfiguredApi(configUrl: string): Promise<string> {
  if (configUrl === configCacheUrl && Date.now() - configCacheAt < CONFIG_CACHE_MS && configCacheValue) {
    return configCacheValue;
  }
  if (configFetch) return configFetch;

  configFetch = (async () => {
    const freshUrl = new URL(configUrl);
    // Defeat intermediary caches so a restarted Kaggle tunnel is picked up promptly.
    freshUrl.searchParams.set("_saer", String(Math.floor(Date.now() / CONFIG_CACHE_MS)));
    const response = await fetch(freshUrl.toString(), {
      cache: "no-store",
      headers: configUrl.includes("api.github.com/gists/") ? { Accept: "application/vnd.github+json" } : undefined,
    });
    if (!response.ok) throw new Error(`تعذر جلب إعداد API (${response.status}).`);
    const value = apiUrlFromPayload(await response.json());
    if (!value) throw new Error("ملف API الخارجي لا يحتوي api_base_url صالحاً.");
    configCacheUrl = configUrl;
    configCacheAt = Date.now();
    configCacheValue = value;
    if (typeof window !== "undefined") window.sessionStorage.setItem(API_SESSION_KEY, value);
    return value;
  })().finally(() => {
    configFetch = undefined;
  });
  return configFetch;
}

export async function resolveApiBaseUrl(): Promise<string> {
  const explicit = queryApiUrl();
  if (explicit) {
    if (typeof window !== "undefined") window.sessionStorage.setItem(API_SESSION_KEY, explicit);
    return explicit;
  }

  const configUrl = getApiConfigUrl();
  if (configUrl) {
    try {
      return await fetchConfiguredApi(configUrl);
    } catch (error) {
      const saved = typeof window === "undefined"
        ? ""
        : normalizeApiBaseUrl(window.sessionStorage.getItem(API_SESSION_KEY) || "") || "";
      if (saved) return saved;
      throw error;
    }
  }
  return getApiBaseUrl();
}

// احتفظ بخيارات الربط عند انتقال SPA إلى مسار لا يحمل معاملات الرابط.
if (typeof window !== "undefined") {
  getApiBaseUrl();
  getApiConfigUrl();
}
