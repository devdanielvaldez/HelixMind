import { createCipheriv, createDecipheriv, createHash, randomBytes } from "node:crypto";

export const COOKIE_NAME = "medai_browser_session";
export const SESSION_MAX_AGE_SECONDS = 8 * 60 * 60;
const VERSION = "v1";
const ALLOWED_ROLES = new Set(["reader", "researcher", "tenant_admin", "platform_admin", "service"]);

export function isValidBrowserIdentity(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const keys = Object.keys(value).sort();
  return keys.length === 3 && keys[0] === "role" && keys[1] === "subject" && keys[2] === "tenant_id"
    && typeof value.subject === "string" && value.subject.length > 0 && value.subject.length <= 128
    && typeof value.tenant_id === "string" && value.tenant_id.length > 0 && value.tenant_id.length <= 128
    && typeof value.role === "string" && ALLOWED_ROLES.has(value.role);
}

function encryptionKey(secret) {
  if (typeof secret !== "string" || Buffer.byteLength(secret, "utf8") < 32) {
    throw new Error("MEDAI_WEB_SESSION_SECRET must contain at least 32 UTF-8 bytes");
  }
  return createHash("sha256").update("medai-browser-session-v1\0").update(secret, "utf8").digest();
}

function validSession(session, now) {
  return session && typeof session === "object"
    && typeof session.token === "string" && session.token.length >= 16 && session.token.length <= 4096
    && isValidBrowserIdentity({ subject: session.subject, tenant_id: session.tenant_id, role: session.role })
    && Number.isSafeInteger(session.expires_at) && session.expires_at > now
    && session.expires_at <= now + SESSION_MAX_AGE_SECONDS;
}

export function encryptBrowserSession(session, secret, now = Math.floor(Date.now() / 1000)) {
  if (!validSession(session, now)) throw new Error("browser session payload is invalid or expired");
  const nonce = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", encryptionKey(secret), nonce);
  cipher.setAAD(Buffer.from(VERSION, "ascii"));
  const ciphertext = Buffer.concat([cipher.update(JSON.stringify(session), "utf8"), cipher.final()]);
  return [VERSION, nonce.toString("base64url"), cipher.getAuthTag().toString("base64url"), ciphertext.toString("base64url")].join(".");
}

export function decryptBrowserSession(value, secret, now = Math.floor(Date.now() / 1000)) {
  if (typeof value !== "string" || value.length > 8192) return null;
  const [version, nonceText, tagText, ciphertextText, extra] = value.split(".");
  if (version !== VERSION || !nonceText || !tagText || !ciphertextText || extra !== undefined) return null;
  try {
    const nonce = Buffer.from(nonceText, "base64url");
    const tag = Buffer.from(tagText, "base64url");
    const ciphertext = Buffer.from(ciphertextText, "base64url");
    if (nonce.length !== 12 || tag.length !== 16 || ciphertext.length > 6144) return null;
    const decipher = createDecipheriv("aes-256-gcm", encryptionKey(secret), nonce);
    decipher.setAAD(Buffer.from(VERSION, "ascii"));
    decipher.setAuthTag(tag);
    const plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()]).toString("utf8");
    const session = JSON.parse(plaintext);
    return validSession(session, now) ? session : null;
  } catch {
    return null;
  }
}
