import type { NextRequest } from "next/server";
import { COOKIE_NAME, decryptBrowserSession } from "@/lib/browser-session.mjs";

export const dynamic = "force-dynamic";

type RouteContext = { params: Promise<{ path: string[] }> };

async function proxy(request: NextRequest, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  if (!path.length || path[0] !== "v1" || path.some((part) => !part || part === "." || part === "..")) {
    return Response.json({ error: "invalid API path" }, { status: 404 });
  }
  if (!["GET", "POST", "PATCH", "PUT", "DELETE"].includes(request.method)) {
    return Response.json({ error: "method not allowed" }, { status: 405, headers: { Allow: "GET, POST, PATCH, PUT, DELETE" } });
  }
  if (request.method !== "GET" && request.headers.get("origin") !== request.nextUrl.origin) {
    return Response.json({ error: "same-origin request required" }, { status: 403 });
  }

  let session;
  try {
    session = decryptBrowserSession(request.cookies.get(COOKIE_NAME)?.value, process.env.MEDAI_WEB_SESSION_SECRET ?? "");
  } catch {
    return Response.json({ error: "browser sessions are not configured" }, { status: 503 });
  }
  if (!session) return Response.json({ error: "login required" }, { status: 401, headers: { "cache-control": "no-store" } });

  const apiBase = process.env.MEDAI_INTERNAL_API_URL ?? "http://127.0.0.1:8090";
  const target = new URL(`/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`, apiBase);
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  headers.set("authorization", `Bearer ${session.token}`);

  let body: Blob | undefined;
  if (request.method !== "GET") {
    const declaredLength = Number(request.headers.get("content-length") ?? "0");
    if (Number.isFinite(declaredLength) && declaredLength > 1_048_576) {
      return Response.json({ error: "request body exceeds 1 MiB" }, { status: 413 });
    }
    const chunks: Uint8Array[] = [];
    let bytesRead = 0;
    const reader = request.body?.getReader();
    if (reader) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        bytesRead += value.byteLength;
        if (bytesRead > 1_048_576) {
          await reader.cancel();
          return Response.json({ error: "request body exceeds 1 MiB" }, { status: 413 });
        }
        chunks.push(value);
      }
    }
    body = new Blob(chunks);
  }
  try {
    const upstream = await fetch(target, { method: request.method, headers, body, cache: "no-store", redirect: "manual", signal: request.signal });
    const responseHeaders = new Headers();
    const upstreamContentType = upstream.headers.get("content-type");
    if (upstreamContentType) responseHeaders.set("content-type", upstreamContentType);
    responseHeaders.set("cache-control", "no-store");
    if (upstream.status === 401) {
      const secure = request.nextUrl.protocol === "https:" ? "; Secure" : "";
      responseHeaders.append("set-cookie", `${COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0${secure}`);
    }
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch {
    return Response.json({ error: "MedAI API is unavailable" }, { status: 502 });
  }
}

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const PUT = proxy;
export const DELETE = proxy;
