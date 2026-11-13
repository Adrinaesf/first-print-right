import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('./public/', import.meta.url));
const MAX_BODY_BYTES = 32 * 1024;
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.woff2': 'font/woff2', '.svg': 'image/svg+xml', '.png': 'image/png', '.stl': 'model/stl', '.txt': 'text/plain; charset=utf-8' };
const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { headline: { type: 'string' }, reasons: { type: 'array', items: { type: 'string' } }, tips: { type: 'array', items: { type: 'string' } } },
  required: ['headline', 'reasons', 'tips'],
};

function json(res, code, data) {
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
  res.end(JSON.stringify(data));
}

async function readReport(req) {
  if (!/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] || '')) throw Object.assign(new Error('Use application/json.'), { status: 415 });
  let body = '';
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw Object.assign(new Error('Report is too large.'), { status: 413 });
    body += chunk.toString('utf8');
  }
  let report;
  try { report = JSON.parse(body); } catch { throw Object.assign(new Error('Invalid JSON.'), { status: 400 }); }
  const metricNames = ['supportG', 'partG', 'totalG', 'hours', 'kwh', 'overhangCm2', 'contactCm2', 'heightMm'];
  if (!report || typeof report !== 'object' || typeof report.rotation_applied !== 'boolean' || !Number.isInteger(report.orientations_tested) || report.orientations_tested < 1) {
    throw Object.assign(new Error('Invalid analysis report.'), { status: 400 });
  }
  const metrics = {};
  for (const key of ['before', 'after']) {
    const values = report[key];
    if (!values || !metricNames.every(name => Number.isFinite(values[name]) && values[name] >= 0) || !['Low', 'Medium', 'High'].includes(values.risk) || typeof values.fits !== 'boolean') {
      throw Object.assign(new Error('Invalid analysis metrics.'), { status: 400 });
    }
    metrics[key] = Object.fromEntries([...metricNames, 'risk', 'fits'].map(name => [name, values[name]]));
  }
  // Whitelist measured values; filenames, geometry and arbitrary text never leave the server.
  return { ...metrics, orientations_tested: report.orientations_tested, rotation_applied: report.rotation_applied };
}

export function createAppServer({ env = process.env, fetchImpl = fetch, publicDir = ROOT } = {}) {
  const apiKey = env.OPENAI_API_KEY?.trim();
  const model = env.OPENAI_MODEL || 'gpt-4.1-mini';
  const effort = env.OPENAI_REASONING_EFFORT || '';
  let inFlight = 0;
  return createServer(async (req, res) => {
    try {
      const path = new URL(req.url, 'http://localhost').pathname;
      if (path === '/api/status') {
        if (req.method !== 'GET') return json(res, 405, { error: 'Method not allowed.' });
        return json(res, 200, { ai: Boolean(apiKey), model, effort: effort || 'default' });
      }
      if (path === '/api/explain') {
        if (req.method !== 'POST') return json(res, 405, { error: 'Method not allowed.' });
        // The browser provides a deterministic summary when no key is configured.
        if (!apiKey) return json(res, 503, { error: 'AI is off. Configure OPENAI_API_KEY on the server.' });
        if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}` && req.headers.origin !== `https://${req.headers.host}`) {
          return json(res, 403, { error: 'Cross-origin requests are not allowed.' });
        }
        const report = await readReport(req);
        if (inFlight >= 2) return json(res, 429, { error: 'AI is busy. Please try again shortly.' });
        inFlight++;
        try {
          const response = await fetchImpl('https://api.openai.com/v1/chat/completions', {
            method: 'POST', signal: AbortSignal.timeout(30_000),
            headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model, ...(effort ? { reasoning_effort: effort } : {}),
              messages: [
                { role: 'system', content: 'Explain an STL print orientation analysis in plain language. Return JSON matching the schema. Use only the supplied measurements; these are rough estimates, not slicer results. Explain support plastic, stability and time changes accurately, including increases. Never promise a successful print. Keep headline to one sentence, reasons to 2-4 short sentences, and tips to 0-2 brief practical tips. Assume PLA, 0.2 mm layers, 15% infill, a 220 x 220 x 250 mm printer. Do not claim to have seen the mesh.' },
                { role: 'user', content: JSON.stringify(report) },
              ],
              response_format: { type: 'json_schema', json_schema: { name: 'print_explanation', strict: true, schema: SCHEMA } },
            }),
          });
          if (!response.ok) return json(res, 502, { error: 'AI service unavailable. Using the local summary.' });
          const data = await response.json();
          const answer = JSON.parse(data.choices?.[0]?.message?.content || 'null');
          if (!answer || typeof answer.headline !== 'string' || !Array.isArray(answer.reasons) || !answer.reasons.every(x => typeof x === 'string') || !Array.isArray(answer.tips) || !answer.tips.every(x => typeof x === 'string')) throw new Error('Invalid AI response');
          return json(res, 200, answer);
        } catch {
          return json(res, 502, { error: 'AI service unavailable. Using the local summary.' });
        } finally { inFlight--; }
      }
      if (path.startsWith('/api/')) return json(res, 404, { error: 'Not found.' });
      if (!['GET', 'HEAD'].includes(req.method)) return json(res, 405, { error: 'Method not allowed.' });
      let decoded;
      try { decoded = decodeURIComponent(path); } catch { return json(res, 400, { error: 'Invalid path.' }); }
      if (decoded.includes('\0') || decoded.split(/[\\/]/).some(part => part.startsWith('.'))) return json(res, 404, { error: 'Not found.' });
      const base = await realpath(publicDir);
      const target = await realpath(resolve(base, '.' + (decoded === '/' ? '/index.html' : decoded)));
      if (!target.startsWith(base + sep)) return json(res, 404, { error: 'Not found.' });
      const info = await stat(target);
      if (!info.isFile()) return json(res, 404, { error: 'Not found.' });
      res.writeHead(200, { 'Content-Type': TYPES[extname(target)] || 'application/octet-stream', 'Content-Length': info.size, 'X-Content-Type-Options': 'nosniff' });
      if (req.method === 'HEAD') return res.end();
      const stream = createReadStream(target);
      stream.on('error', () => res.destroy());
      stream.pipe(res);
    } catch (err) {
      if (!res.headersSent) json(res, err.status || (err.code === 'ENOENT' || err.code === 'ENOTDIR' ? 404 : 500), { error: err.status ? err.message : err.code === 'ENOENT' || err.code === 'ENOTDIR' ? 'Not found.' : 'Request failed.' });
      else res.destroy();
    }
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 3000);
  const host = process.env.HOST || '127.0.0.1';
  const server = createAppServer();
  server.listen(port, host, () => console.log(`First-Print-Right: http://${host}:${server.address().port}`));
}
