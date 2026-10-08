import assert from "node:assert/strict";
import test from "node:test";

import {
  decryptBrowserSession,
  encryptBrowserSession,
  isValidBrowserIdentity,
  SESSION_MAX_AGE_SECONDS,
} from "../src/lib/browser-session.mjs";

const secret = "test-only-session-encryption-key-0123456789";
const now = 1_800_000_000;
const identity = {
  token: "researcher-secret-bearer-token-value",
  subject: "researcher-17",
  tenant_id: "lab-a",
  role: "researcher",
  expires_at: now + SESSION_MAX_AGE_SECONDS,
};

test("browser session is encrypted and decrypts to its authenticated identity", () => {
  const cookie = encryptBrowserSession(identity, secret, now);
  assert.match(cookie, /^v1\./);
  assert.equal(cookie.includes(identity.token), false);
  assert.deepEqual(decryptBrowserSession(cookie, secret, now), identity);
});

test("browser session rejects tampering, wrong keys, and expiry", () => {
  const cookie = encryptBrowserSession(identity, secret, now);
  const parts = cookie.split(".");
  parts[3] = `${parts[3].slice(0, 5)}${parts[3][5] === "A" ? "B" : "A"}${parts[3].slice(6)}`;
  assert.equal(decryptBrowserSession(parts.join("."), secret, now), null);
  assert.equal(decryptBrowserSession(cookie, `${secret}-wrong`, now), null);
  assert.equal(decryptBrowserSession(cookie, secret, identity.expires_at), null);
});

test("browser session requires a high-entropy configured secret and valid identity", () => {
  assert.throws(() => encryptBrowserSession(identity, "short", now), /at least 32/);
  assert.throws(() => encryptBrowserSession({ ...identity, role: "service_admin" }, secret, now), /invalid/);
});

test("API identity validation accepts only complete supported identities", () => {
  assert.equal(isValidBrowserIdentity({ subject: "researcher-1", tenant_id: "lab-1", role: "researcher" }), true);
  assert.equal(isValidBrowserIdentity({ subject: "", tenant_id: "lab-1", role: "researcher" }), false);
  assert.equal(isValidBrowserIdentity({ subject: "researcher-1", tenant_id: "lab-1", role: "root" }), false);
  assert.equal(isValidBrowserIdentity({ subject: "researcher-1", tenant_id: "lab-1", role: "researcher", bearer: "secret" }), false);
  assert.equal(isValidBrowserIdentity(null), false);
});

import { officialTrialHistoryUrl } from "../src/lib/trial-history.mjs";

test("official trial history links accept only exact NCT identifiers", () => {
  assert.equal(officialTrialHistoryUrl("NCT01234567"), "https://clinicaltrials.gov/study/NCT01234567?tab=history");
  assert.equal(officialTrialHistoryUrl("NCT0123456"), null);
  assert.equal(officialTrialHistoryUrl("javascript:alert(1)"), null);
  assert.equal(officialTrialHistoryUrl("NCT01234567?next=evil"), null);
  assert.equal(officialTrialHistoryUrl(null), null);
});
