import type { NextRequest } from "next/server";
import type { BrowserIdentity, BrowserSession } from "@/lib/browser-session.mjs";
import {
  COOKIE_NAME,
  SESSION_MAX_AGE_SECONDS,
  decryptBrowserSession,
  encryptBrowserSession,
  isValidBrowserIdentity,
} from "@/lib/browser-session.mjs";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_LOGIN_BODY_BYTES = 8192;

function cookieHeader(maxAge: number, request: NextRequest): string {
  const secure = request.nextUrl.protocol === "https:" ? "; Secure" : "";
  return `${COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=${maxAge}${secure}`;
}

function json(data: unknown, status = 200, headers?: HeadersInit): Response {
  const resultHeaders = new Headers(headers);
  resultHeaders.set("content-type", "application/json; charset=utf-8");
  resultHeaders.set("cache-control", "no-store");
  return new Response(JSON.stringify(data), { status, headers: resultHeaders });
}

async function boundedText(request: NextRequest): Promise<string | null> {
  const declared = Number(request.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > MAX_LOGIN_BODY_BYTES) return null;
  const reader = request.body?.getReader();
  if (!reader) return "";
  const chunks: Uint8Array[] = [];
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > MAX_LOGIN_BODY_BYTES) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}

function configuredApiUrl(): URL {
  const url = new URL(process.env.MEDAI_INTERNAL_API_URL ?? "http://127.0.0.1:8090");
  if (url.username || url.password || url.search || url.hash || !["http:", "https:"].includes(url.protocol)) {
    throw new Error("invalid internal API URL");
  }
  return url;
}

export async function GET(request: NextRequest): Promise<Response> {
  let session;
  try {
    session = decryptBrowserSession(request.cookies.get(COOKIE_NAME)?.value, process.env.MEDAI_WEB_SESSION_SECRET ?? "");
  } catch {
    return json({ error: "browser sessions are not configured" }, 503, { "set-cookie": cookieHeader(0, request) });
  }
  if (!session) return json({ error: "login required" }, 401, { "set-cookie": cookieHeader(0, request) });
  return json({ subject: session.subject, tenant_id: session.tenant_id, role: session.role, expires_at: session.expires_at });
}

export async function POST(request: NextRequest): Promise<Response> {
  if (request.headers.get("origin") !== request.nextUrl.origin) return json({ error: "same-origin request required" }, 403);
  const raw = await boundedText(request).catch(() => null);
  if (raw === null) return json({ error: "login request exceeds 8 KiB" }, 413);
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    return json({ error: "invalid login request" }, 400);
  }
  if (!body || typeof body !== "object" || Array.isArray(body) || Object.keys(body).some((key) => key !== "token")) {
    return json({ error: "login request must contain only a bearer token" }, 400);
  }
  const token = (body as { token?: unknown }).token;
  if (typeof token !== "string" || token.length < 16 || token.length > 4096 || /[\r\n]/.test(token)) {
    return json({ error: "invalid API credential" }, 400);
  }
  let identityResponse: Response;
  try {
    const endpoint = new URL("/v1/auth/me", configuredApiUrl());
    identityResponse = await fetch(endpoint, {
      headers: { authorization: `Bearer ${token}`, accept: "application/json" },
      cache: "no-store",
      redirect: "manual",
      signal: AbortSignal.timeout(8000),
    });
  } catch {
    return json({ error: "MedAI API is unavailable" }, 502);
  }
  if (identityResponse.status === 401 || identityResponse.status === 403) return json({ error: "credential was not accepted" }, 401);
  if (!identityResponse.ok || identityResponse.status >= 300) return json({ error: "MedAI API could not validate the credential" }, 502);
  let identity: BrowserIdentity;
  try {
    const text = await identityResponse.text();
    if (new TextEncoder().encode(text).byteLength > MAX_LOGIN_BODY_BYTES) return json({ error: "identity response is too large" }, 502);
    const parsed: unknown = JSON.parse(text);
    if (!isValidBrowserIdentity(parsed)) return json({ error: "MedAI API returned an invalid identity" }, 502);
    identity = parsed;
  } catch {
    return json({ error: "MedAI API returned an invalid identity" }, 502);
  }
  const expiresAt = Math.floor(Date.now() / 1000) + SESSION_MAX_AGE_SECONDS;
  const cookieSession: BrowserSession = { token, subject: identity.subject, tenant_id: identity.tenant_id, role: identity.role, expires_at: expiresAt };
  try {
    const encrypted = encryptBrowserSession(cookieSession, process.env.MEDAI_WEB_SESSION_SECRET ?? "");
    const secure = request.nextUrl.protocol === "https:" ? "; Secure" : "";
    const setCookie = `${COOKIE_NAME}=${encrypted}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${SESSION_MAX_AGE_SECONDS}${secure}`;
    return json({ subject: identity.subject, tenant_id: identity.tenant_id, role: identity.role, expires_at: expiresAt }, 200, { "set-cookie": setCookie });
  } catch {
    return json({ error: "browser sessions are not configured" }, 503);
  }
}

export async function DELETE(request: NextRequest): Promise<Response> {
  if (request.headers.get("origin") !== request.nextUrl.origin) return json({ error: "same-origin request required" }, 403);
  return json({ logged_out: true }, 200, { "set-cookie": cookieHeader(0, request) });
}
