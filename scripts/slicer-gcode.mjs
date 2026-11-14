/** Read plain-text Prusa G-code; support labels are deposited filament only. */
export function parseGcode(text) {
  const match = text.match(/; estimated printing time \(normal mode\) = ([^\r\n]+)/);
  if (!match) throw new Error('Missing normal-mode print time in G-code.');
  const units = { d: 86400, h: 3600, m: 60, s: 1 };
  let seconds = 0;
  for (const [, amount, unit] of match[1].matchAll(/(\d+(?:\.\d+)?)\s*([dhms])/g)) seconds += Number(amount) * units[unit];
  if (!(seconds > 0)) throw new Error('Invalid print duration.');
  let relative = false, ePosition = 0, support = false, length = 0;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith(';TYPE:')) support = /^;TYPE:Support material(?: interface)?\s*$/.test(line);
    const code = line.split(';')[0].trim();
    if (/^M83\b/.test(code)) relative = true;
    if (/^M82\b/.test(code)) relative = false;
    const e = code.match(/\bE(-?(?:\d+(?:\.\d*)?|\.\d+))/);
    if (!e) continue;
    const value = Number(e[1]);
    if (/^G92\b/.test(code)) { ePosition = value; continue; }
    if (!/^G[01]\b/.test(code)) continue;
    const delta = relative ? value : value - ePosition;
    ePosition = relative ? ePosition + value : value;
    // Stationary unretracts restore pressure; only XY extrusion deposits material.
    if (support && delta > 0 && /\b[XY]-?(?:\d|\.\d)/.test(code)) length += delta;
  }
  return { supportG: length * Math.PI * (1.75/2)**2 * 1.24 / 1000, hours: seconds / 3600 };
}
