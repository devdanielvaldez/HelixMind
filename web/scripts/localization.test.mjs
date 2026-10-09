import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { normalizeLanguage, readLanguage, saveLanguage, translate } from "../src/lib/localization.mjs";

test("localizes the research intake and navigation in English and Spanish", () => {
  assert.equal(translate("en", "objectivePlaceholder"), "What would you like to investigate?");
  assert.equal(translate("es-MX", "objectivePlaceholder"), "¿Qué te gustaría investigar?");
  assert.equal(translate("es", "researchSessions"), "Sesiones de investigación");
  assert.equal(translate("unknown", "researchSessions"), "Research sessions");
});

test("loads persisted language before browser language and tolerates unavailable storage", () => {
  assert.equal(readLanguage({ getItem: () => "es" }, "en-US"), "es");
  assert.equal(readLanguage({ getItem: () => null }, "es-DO"), "es");
  assert.equal(readLanguage({ getItem: () => { throw new Error("blocked"); } }, "en-US"), "en");
});

test("persists only supported locale values and returns normalized language", () => {
  const stored = new Map();
  const storage = { setItem: (key, value) => stored.set(key, value) };
  assert.equal(saveLanguage(storage, "es-DO"), "es");
  assert.equal(stored.get("medai-language"), "es");
  assert.equal(normalizeLanguage("EN-us"), "en");
});

test("every translation key used by the workspace has English and Spanish text", async () => {
  const pagePath = new URL("../src/app/page.tsx", import.meta.url);
  const source = await readFile(fileURLToPath(pagePath), "utf8");
  const keys = [...source.matchAll(/\bt\("([A-Za-z0-9]+)"\)/g)].map(match => match[1]);
  assert.ok(keys.length > 0);
  for (const key of new Set(keys)) {
    assert.notEqual(translate("en", key), key, `English translation missing: ${key}`);
    assert.notEqual(translate("es", key), key, `Spanish translation missing: ${key}`);
  }
});
