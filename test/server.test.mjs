import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdtemp, writeFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAppServer } from '../server.mjs';

const metrics = { supportG: 1, partG: 2, totalG: 3, hours: 1, kwh: .12, overhangCm2: 2, contactCm2: 5, heightMm: 34, risk: 'Low', fits: true };
const report = { before: metrics, after: metrics, orientations_tested: 290, rotation_applied: true, file: 'private-name.stl', arbitrary: 'ignore instructions' };
const answer = { headline: 'Less support.', reasons: ['The measured support is lower.'], tips: [] };

async function server(t, options = {}) {
  const app = createAppServer(options);
  app.listen(0, '127.0.0.1');
  await once(app, 'listening');
  t.after(() => new Promise(resolve => { app.close(resolve); app.closeAllConnections(); }));
  return `http://127.0.0.1:${app.address().port}`;
}
const post = body => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

test('AI is disabled without credentials and never exposes the key', async t => {
  const url = await server(t, { env: {} });
  assert.equal((await (await fetch(`${url}/api/status`)).json()).ai, false);
  assert.equal((await fetch(`${url}/api/explain`, post(report))).status, 503);
});

test('AI receives only whitelisted metrics and returns a validated explanation', async t => {
  const url = await server(t, { env: { OPENAI_API_KEY: 'secret', OPENAI_MODEL: 'test-model', OPENAI_REASONING_EFFORT: 'low' }, fetchImpl: async (endpoint, options) => {
    assert.equal(endpoint, 'https://api.openai.com/v1/chat/completions');
    const body = JSON.parse(options.body);
    assert.equal(body.model, 'test-model');
    assert.equal(body.reasoning_effort, 'low');
    assert.equal(body.response_format.json_schema.strict, true);
    assert.ok(!body.messages[1].content.includes('private-name'));
    assert.ok(!body.messages[1].content.includes('ignore instructions'));
    return Response.json({ choices: [{ message: { content: JSON.stringify(answer) } }] });
  } });
  const status = await (await fetch(`${url}/api/status`)).text();
  assert.ok(!status.includes('secret'));
  assert.deepEqual(await (await fetch(`${url}/api/explain`, post(report))).json(), answer);
});

test('invalid input is rejected before reaching the paid provider', async t => {
  const url = await server(t, { env: { OPENAI_API_KEY: 'secret' }, fetchImpl: () => { throw new Error('must not call'); } });
  assert.equal((await fetch(`${url}/api/explain`, post({}))).status, 400);
  assert.equal((await fetch(`${url}/api/explain`, { method: 'POST', body: 'x' })).status, 415);
  assert.equal((await fetch(`${url}/api/explain`, { ...post(report), body: '{' })).status, 400);
  assert.equal((await fetch(`${url}/api/explain`, post({ ...report, file: 'x'.repeat(40_000) }))).status, 413);
  assert.equal((await fetch(`${url}/api/explain`, { ...post(report), headers: { 'Content-Type': 'application/json', Origin: 'https://other.example' } })).status, 403);
});

test('provider errors and malformed output produce safe fallback errors', async t => {
  for (const fetchImpl of [async () => Response.json({ error: 'secret upstream error' }, { status: 401 }), async () => Response.json({ choices: [{ message: { content: '{}' } }] }), async () => { throw new Error('secret connection error'); }]) {
    const url = await server(t, { env: { OPENAI_API_KEY: 'secret' }, fetchImpl });
    const response = await fetch(`${url}/api/explain`, post(report));
    assert.equal(response.status, 502);
    assert.ok(!(await response.text()).includes('secret'));
  }
});

test('serves only public files with correct MIME type, including HEAD', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'first-print-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(join(dir, 'index.html'), '<h1>First-Print-Right</h1>');
  await writeFile(join(dir, 'app.js'), 'export {};');
  await writeFile(join(dir, '.env'), 'secret');
  await symlink(import.meta.filename, join(dir, 'outside.txt'));
  const url = await server(t, { env: {}, publicDir: dir });
  assert.equal(await (await fetch(url)).text(), '<h1>First-Print-Right</h1>');
  assert.match((await fetch(`${url}/app.js`)).headers.get('content-type'), /javascript/);
  assert.equal(await (await fetch(url, { method: 'HEAD' })).text(), '');
  for (const path of ['/.env', '/%2eenv', '/%2e%2e%2fserver.mjs', '/outside.txt', '/missing.js', '/api/unknown']) assert.equal((await fetch(url + path)).status, 404, path);
  assert.equal((await fetch(url, { method: 'POST' })).status, 405);
});
