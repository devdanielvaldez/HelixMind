import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const path = process.argv[2];
if (!path) throw new Error("usage: node verify-compose-network-isolation.mjs <compose-config.json>");
const config = JSON.parse(readFileSync(path, "utf8"));
const networks = config.networks;
const services = config.services;
assert(networks && services, "rendered Compose config must contain networks and services");

function networkEndingIn(suffix) {
  const name = Object.keys(networks).find((candidate) => candidate.endsWith(suffix));
  assert(name, `missing configured network ${suffix}`);
  return name;
}

const backend = networkEndingIn("api-backend");
const metrics = networkEndingIn("metrics");
const egress = networkEndingIn("provider-egress");
assert.equal(networks[backend].internal, true, "API backend must not provide external routing");
assert.equal(networks[metrics].internal, true, "metrics network must not provide external routing");
assert.notEqual(networks[egress].internal, true, "provider egress network must allow outbound provider calls");

function attached(serviceName) {
  const service = services[serviceName];
  assert(service, `missing service ${serviceName}`);
  return Object.keys(service.networks ?? {});
}

function expectNetworks(serviceName, expected) {
  assert.deepEqual(attached(serviceName).sort(), expected.slice().sort(), `${serviceName} network isolation changed`);
}

expectNetworks("api", [backend, metrics]);
expectNetworks("web", [backend]);
expectNetworks("science-mcp", [backend, egress]);
expectNetworks("science-worker", [backend, egress]);
expectNetworks("science-specialist-worker", [backend, egress]);
expectNetworks("science-direct-worker", [backend]);
expectNetworks("science-direct-specialist-worker", [backend]);
expectNetworks("prometheus", [metrics]);
expectNetworks("alertmanager", [metrics, egress]);
console.log("Compose network isolation policy verified.");
