'use strict';

/* ============================================================
   Utilidades
   ============================================================ */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (v, d = 0) => (v == null ? '—' : `${(v * 100).toFixed(d)} %`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

let COL = {};
const readColors = () => { COL = { ok: cssVar('--ok'), bad: cssVar('--bad'), warn: cssVar('--warn'), neutral: cssVar('--neutral'), dmg: cssVar('--dmg') }; };

const ROUTE_KEYS = ['empaque', 'industria', 'descarte', 'revision'];
const ROUTE_CSS = { empaque: 'var(--ok)', industria: 'var(--warn)', descarte: 'var(--bad)', revision: 'var(--neutral)' };
const ROUTE_ICON = { empaque: '✓', industria: '↻', descarte: '✕', revision: '?' };
const CLS_ES = { Crack: 'Rajado', Intact: 'Sano' };
const LEVEL_ES = { leve: 'leve', media: 'media', grave: 'grave' };

const App = { info: null, samples: [], routes: {} };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* sin cuerpo JSON */ }
    throw new Error(msg || `HTTP ${r.status}`);
  }
  return r.json();
}

const imgCache = new Map();
function loadImage(src) {
  if (imgCache.has(src)) return imgCache.get(src);
  const p = new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
  if (!src.startsWith('data:') && !src.startsWith('blob:')) imgCache.set(src, p);
  return p;
}

async function withMasks(res) {
  await Promise.all(res.eggs.filter((e) => e.damage).map(async (e) => {
    [e._mask, e._crop] = await Promise.all([loadImage(e.damage.mask_png), loadImage(e.damage.crop_png)]);
  }));
  return res;
}

function routePill(route, lg = false) {
  const r = App.routes[route] || { name: route };
  return `<span class="route-pill ${lg ? 'lg' : ''}" style="--c:${ROUTE_CSS[route]}">${ROUTE_ICON[route] || ''} ${esc(r.name)}</span>`;
}

function sevBar(sev) {
  const w = Math.min(100, (sev || 0) * 100);
  return `<div class="sev" title="Gravedad = píxeles dañados / píxeles del huevo"><b style="width:${w}%"></b><i style="left:15%"></i><i style="left:35%"></i></div>
    <div class="sev-scale"><span>0 %</span><span>leve &lt; 15 % · media &lt; 35 % · grave</span><span>100 %</span></div>`;
}

function worstEgg(res) {
  const cr = res.eggs.filter((e) => e.damage);
  if (cr.length) return cr.reduce((a, b) => (b.damage.severity > a.damage.severity ? b : a));
  return res.eggs[0] || null;
}

/* Dibuja la imagen + máscaras de daño + cajas en un canvas que se ajusta a su contenedor. */
function drawResult(canvas, img, res, o = {}) {
  const iw = img.naturalWidth || img.videoWidth || img.width;
  const ih = img.naturalHeight || img.videoHeight || img.height;
  if (!iw || !ih) return;
  const boxW = canvas.parentElement.clientWidth || 400;
  const k = Math.min(boxW / iw, (o.maxH || 420) / ih, o.maxUp || 3);
  const W = Math.round(iw * k), H = Math.round(ih * k);
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(img, 0, 0, W, H);
  if (!res) return;
  const s = W / res.width;
  if (o.mask !== false) {
    for (const e of res.eggs) {
      if (!e.damage || !e._mask) continue;
      const [rx, ry, rw, rh] = e.damage.rect;
      ctx.drawImage(e._mask, rx * s, ry * s, rw * s, rh * s);
    }
  }
  if (o.boxes !== false) {
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    res.eggs.forEach((e, i) => {
      const [x1, y1, x2, y2] = e.box.map((v) => v * s);
      const col = e.cls === 0 ? COL.bad : COL.ok;
      ctx.lineWidth = 2.5; ctx.strokeStyle = col;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 6) : ctx.rect(x1, y1, x2 - x1, y2 - y1); ctx.stroke();
      const label = `${res.eggs.length > 1 ? `${i + 1} · ` : ''}${CLS_ES[e.label]} ${Math.round(e.conf * 100)} %${e.damage ? ` · daño ${pct(e.damage.severity)}` : ''}`;
      const tw = ctx.measureText(label).width + 12;
      const ty = y1 - 21 < 0 ? y1 + 3 : y1 - 21;
      const tx = Math.min(Math.max(0, x1 - 1), W - tw);
      ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(tx, ty, tw, 19, 5) : ctx.rect(tx, ty, tw, 19); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.fillText(label, tx + 6, ty + 14);
    });
  }
}

/* Recorte 192×192 que ve dano_v1, con o sin la máscara encima. */
function drawCrop(canvas, egg, withMask) {
  const S = 192;
  canvas.width = S; canvas.height = S;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(egg._crop, 0, 0, S, S);
  if (withMask) ctx.drawImage(egg._mask, 0, 0, S, S);
}

function cropPair(egg) {
  const wrap = document.createElement('div');
  wrap.className = 'mini';
  wrap.innerHTML = '<figure><canvas></canvas><figcaption>Recorte 192×192</figcaption></figure><figure><canvas></canvas><figcaption>Zona dañada</figcaption></figure>';
  const [a, b] = $$('canvas', wrap);
  drawCrop(a, egg, false); drawCrop(b, egg, true);
  return wrap;
}

/* ============================================================
   Pestañas
   ============================================================ */
function initTabs() {
  $$('.tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
  const fromHash = () => { const h = location.hash.replace('#', ''); if (h && $(`#tab-${h}`)) showTab(h); };
  window.addEventListener('hashchange', fromHash);
  fromHash();
}
function showTab(name) {
  $$('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
  history.replaceState(null, '', `#${name}`);
  if (name !== 'banda') Belt.pause();
  if (name === 'banda') Belt.layout();
  if (name !== 'vivo') Live.stop();
  if (name === 'analizar') Analyze.redraw();
}

/* ============================================================
   Banda transportadora
   ============================================================ */
const EGG_W = 56, EGG_H = 68, SPACING = 118;
const SPEEDS = [0, 45, 70, 100, 135, 175]; // px/s

const Belt = {
  running: false, s: 0, lastT: 0, lastSpawnS: -1e9, eggs: [], seq: 0, runTime: 0,
  bags: {}, geom: null, stats: null, pending: 0,

  init() {
    this.stage = $('#stage'); this.layer = $('#eggLayer'); this.beltEl = $('#belt'); this.beam = $('#beam');
    $('#beltStart').onclick = () => this.start();
    $('#beltPause').onclick = () => this.pause();
    $('#beltReset').onclick = () => this.reset();
    $('#beltGt').onchange = () => this.stage.classList.toggle('hide-gt', !$('#beltGt').checked);
    $('#beltMix').onchange = () => { this.bags = {}; };
    this.buildBins();
    this.resetStats();
    window.addEventListener('resize', () => this.layout());
    this.layout();
    requestAnimationFrame((t) => this.tick(t));
  },

  buildBins() {
    const bins = $('#bins');
    bins.innerHTML = ROUTE_KEYS.map((k) => `
      <div class="gate" data-gate="${k}" style="--c:${ROUTE_CSS[k]}"></div>
      <div class="bin" data-bin="${k}" style="--c:${ROUTE_CSS[k]}">
        <div class="pile"></div><div class="count">0</div>
        <div class="name">${ROUTE_ICON[k]} ${esc(App.routes[k]?.name || k)}</div>
        <div class="rule">${esc(App.routes[k]?.rule || '')}</div>
      </div>`).join('');
  },

  layout() {
    if (!this.stage) return;
    const W = this.stage.clientWidth;
    if (!W) return;
    const binW = Math.min(170, W * 0.155);
    const fx = { empaque: 0.43, industria: 0.60, descarte: 0.77, revision: 0.915 };
    this.geom = { W, startX: 44, camX: Math.max(130, W * 0.2), binX: {}, binW };
    for (const k of ROUTE_KEYS) {
      const x = Math.round(W * fx[k]);
      this.geom.binX[k] = x;
      const bin = $(`[data-bin="${k}"]`); bin.style.left = `${x}px`; bin.style.width = `${binW}px`;
      $(`[data-gate="${k}"]`).style.left = `${x}px`;
    }
    $('#camera').style.left = `${this.geom.camX}px`;
  },

  resetStats() {
    this.stats = { n: 0, correct: 0, latSum: 0, rtSum: 0, routed: 0, bins: Object.fromEntries(ROUTE_KEYS.map((k) => [k, 0])),
      conf: { Crack: { Crack: 0, Intact: 0, Ninguno: 0 }, Intact: { Crack: 0, Intact: 0, Ninguno: 0 } }, log: [] };
    this.runTime = 0;
    this.renderStats();
  },

  reset() {
    this.pause();
    this.eggs.forEach((e) => e.el.remove());
    this.eggs = []; this.s = 0; this.lastSpawnS = -1e9; this.seq = 0;
    $$('.bin').forEach((b) => { $('.count', b).textContent = '0'; $('.pile', b).innerHTML = ''; });
    this.resetStats();
    $('#inspInfo').innerHTML = '<p class="muted">Todavía no ha pasado ningún huevo por la cámara.</p>';
    const c = $('#inspCanvas'); c.width = 10; c.height = 10; c.style.width = ''; c.style.height = '';
    $('#inspMeta').textContent = '';
    $('#stageHint').hidden = false;
  },

  start() {
    if (!App.samples.length) { alert('El servidor no tiene imágenes de prueba configuradas (EGGS_SAMPLES_DIR).'); return; }
    this.layout();
    this.running = true;
    $('#stageHint').hidden = true;
  },
  pause() { this.running = false; this.beam?.classList.remove('wait'); },

  pool(mix) {
    const S = App.samples;
    if (mix === 'crack') return S.filter((s) => s.gt === 0);
    if (mix === 'intact') return S.filter((s) => s.gt === 1);
    if (mix === 'montaje') return S.filter((s) => s.source === 'montaje');
    return S;
  },
  draw(key, list) { // bolsa barajada: no repite hasta agotar
    if (!this.bags[key] || !this.bags[key].length) this.bags[key] = shuffle(list.slice());
    return this.bags[key].pop();
  },
  pick() {
    const mix = $('#beltMix').value;
    if (mix === 'balanced') { // bloques de 4 (2 rajados + 2 sanos) para que la mezcla no tenga rachas
      const g = this.draw('gt', [0, 0, 1, 1]);
      return this.draw(`b${g}`, App.samples.filter((s) => s.gt === g));
    }
    return this.draw(mix, this.pool(mix));
  },

  spawn() {
    const sample = this.pick();
    if (!sample) return;
    const el = document.createElement('div');
    el.className = 'egg';
    const [cx, cy, bw, bh] = sample.box;
    const k = Math.max(EGG_W / (bw * sample.w), EGG_H / (bh * sample.h)) * 0.97;
    el.style.backgroundImage = `url(/api/samples/${sample.id}/image)`;
    el.style.backgroundSize = `${sample.w * k}px ${sample.h * k}px`;
    el.style.backgroundPosition = `${EGG_W / 2 - cx * sample.w * k}px ${EGG_H / 2 - cy * sample.h * k}px`;
    el.innerHTML = `<span class="gt">real: ${CLS_ES[sample.gt_label]}</span>`;
    this.layer.appendChild(el);
    const egg = { id: ++this.seq, sample, el, s0: this.s, state: 'moving', res: null, t0: performance.now() };
    this.eggs.push(egg);
    this.lastSpawnS = this.s;
    this.pending++;
    api(`/api/samples/${sample.id}/predict`, { method: 'POST' })
      .then(withMasks)
      .then(async (res) => { egg.img = await loadImage(`/api/samples/${sample.id}/image`); egg.rt = performance.now() - egg.t0; egg.res = res; })
      .catch((err) => { egg.res = { error: String(err.message || err), eggs: [], route: 'revision', summary: { n_eggs: 0, n_crack: 0 }, timing_ms: { total: 0 } }; })
      .finally(() => { this.pending--; });
  },

  xOf(e) { return this.geom.startX + (this.s - e.s0); },

  tick(t) {
    const dt = Math.min(0.05, (t - (this.lastT || t)) / 1000);
    this.lastT = t;
    if (this.running && this.geom) {
      const { camX, binX } = this.geom;
      let ds = SPEEDS[+$('#beltSpeed').value] * dt;
      let waiting = false;
      for (const e of this.eggs) {
        if (e.state !== 'moving' || e.res) continue;
        const x = this.xOf(e);
        if (x + ds >= camX) { ds = Math.max(0, camX - x); waiting = true; } // la banda espera al modelo
      }
      this.beam.classList.toggle('wait', waiting);
      this.s += ds;
      this.runTime += dt;
      if (this.s - this.lastSpawnS >= SPACING && this.pending < 4) this.spawn();
      for (const e of this.eggs) {
        const x = this.xOf(e);
        if (e.state === 'moving' && e.res && x >= camX) this.inspect(e);
        if (e.state === 'routed' && x >= binX[e.res.route]) this.drop(e, x);
      }
      this.beltEl.style.backgroundPosition = `${this.s % 40}px 0`;
    }
    for (const e of this.eggs) if (e.state !== 'dropping') e.el.style.transform = `translateX(${this.xOf(e)}px)`;
    requestAnimationFrame((tt) => this.tick(tt));
  },

  inspect(e) {
    e.state = 'routed';
    const r = e.res, s = e.sample, st = this.stats;
    this.beam.classList.add('flash');
    setTimeout(() => this.beam.classList.remove('flash'), 180);
    const pred = r.summary.n_eggs === 0 ? 'Ninguno' : (r.summary.n_crack ? 'Crack' : 'Intact');
    const ok = pred === s.gt_label;
    st.n++; if (ok) st.correct++;
    st.latSum += r.timing_ms.total; st.rtSum += e.rt || 0;
    st.conf[s.gt_label][pred]++;
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.style.setProperty('--c', ROUTE_CSS[r.route]);
    tag.textContent = pred === 'Intact' ? '✓ Sano' : pred === 'Crack' ? `✕ ${pct(r.summary.max_severity)}` : '? sin huevo';
    e.el.appendChild(tag);
    st.log.unshift({ id: e.id, s, pred, ok, r, rt: e.rt });
    st.log.length = Math.min(st.log.length, 15);
    this.renderInspection(e, pred, ok);
    this.renderStats();
  },

  drop(e, x) {
    e.state = 'dropping';
    const route = e.res.route;
    const gate = $(`[data-gate="${route}"]`);
    gate.classList.add('open');
    e.el.classList.add('drop');
    const dy = this.stage.clientHeight - 104 - EGG_H - 28;
    requestAnimationFrame(() => { e.el.style.transform = `translate(${x}px, ${dy}px) scale(.45)`; e.el.style.opacity = '0.1'; });
    setTimeout(() => {
      gate.classList.remove('open');
      e.el.remove();
      this.eggs = this.eggs.filter((q) => q !== e);
      this.stats.bins[route]++; this.stats.routed++;
      const bin = $(`[data-bin="${route}"]`), cnt = $('.count', bin);
      cnt.textContent = this.stats.bins[route];
      cnt.classList.add('bump'); setTimeout(() => cnt.classList.remove('bump'), 200);
      const pile = $('.pile', bin);
      pile.insertAdjacentHTML('afterbegin', '<i></i>');
      while (pile.children.length > 24) pile.lastChild.remove();
      this.renderStats();
    }, 560);
  },

  renderInspection(e, pred, ok) {
    const r = e.res, s = e.sample;
    if (e.img) drawResult($('#inspCanvas'), e.img, r, { maxH: 300 });
    $('#inspMeta').textContent = `Huevo #${e.id} · ${s.file.slice(0, 28)}…`;
    const w = worstEgg(r);
    const info = $('#inspInfo');
    if (r.error) { info.innerHTML = `<p class="bad-txt">Error del servidor: ${esc(r.error)}</p>${routePill('revision', true)}`; return; }
    info.innerHTML = `
      ${routePill(r.route, true)}
      <p class="small muted" style="margin-top:6px">${esc(App.routes[r.route]?.desc || '')}</p>
      <dl class="kv">
        <dt>Modelo (v2)</dt><dd>${pred === 'Ninguno' ? 'sin huevo detectado' : `${CLS_ES[pred]} · ${Math.round((w?.conf || 0) * 100)} %`}${r.eggs.length > 1 ? ` (${r.eggs.length} huevos)` : ''}</dd>
        <dt>Etiqueta real</dt><dd>${CLS_ES[s.gt_label]} <span class="${ok ? 'ok-txt' : 'bad-txt'}">${ok ? '✓ acierto' : '✗ error'}</span></dd>
        <dt>Zona dañada</dt><dd>${w?.damage ? `${pct(w.damage.severity, 1)} · ${LEVEL_ES[w.damage.level]}` : pred === 'Intact' ? 'no aplica (huevo sano)' : '—'}</dd>
        <dt>Tiempo</dt><dd>${r.timing_ms.detector} ms detector + ${r.timing_ms.damage} ms daño</dd>
      </dl>
      ${w?.damage ? sevBar(w.damage.severity) : ''}`;
    if (w?.damage) info.appendChild(cropPair(w));
  },

  renderStats() {
    const st = this.stats;
    const acc = st.n ? st.correct / st.n : null;
    const thr = this.runTime > 3 ? (st.routed / this.runTime) * 60 : null;
    $('#kpis').innerHTML = [
      ['Huevos inspeccionados', st.n],
      ['Acierto frente a la etiqueta real', acc == null ? '—' : pct(acc, 1)],
      ['Latencia media del modelo', st.n ? `${Math.round(st.latSum / st.n)} ms` : '—'],
      ['Huevos clasificados por minuto', thr == null ? '—' : thr.toFixed(0)],
    ].map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
    const c = st.conf;
    const cell = (real, p) => `<td class="${real === p ? 'hit' : c[real][p] ? 'miss' : ''}">${c[real][p]}</td>`;
    $('#confusion').innerHTML = `
      <thead><tr><th>Real \\ Modelo</th><th>Rajado</th><th>Sano</th><th>Ninguno</th></tr></thead>
      <tbody>
        <tr><th>Rajado</th>${cell('Crack', 'Crack')}${cell('Crack', 'Intact')}${cell('Crack', 'Ninguno')}</tr>
        <tr><th>Sano</th>${cell('Intact', 'Crack')}${cell('Intact', 'Intact')}${cell('Intact', 'Ninguno')}</tr>
      </tbody>`;
    $('#logBody').innerHTML = st.log.map((x) => {
      const w = worstEgg(x.r);
      const sev = w?.damage ? `<span class="bar"><b style="width:${Math.min(100, w.damage.severity * 100)}%"></b></span>${pct(w.damage.severity)}` : '<span class="muted">—</span>';
      return `<tr>
        <td><div class="thumb" style="background-image:url(/api/samples/${x.s.id}/image)"></div></td>
        <td class="num">${x.id}</td>
        <td>${CLS_ES[x.s.gt_label]}</td>
        <td class="${x.ok ? 'ok-txt' : 'bad-txt'}">${x.pred === 'Ninguno' ? 'Ninguno' : `${CLS_ES[x.pred]} ${Math.round((w?.conf || 0) * 100)} %`}</td>
        <td>${sev}</td>
        <td>${routePill(x.r.route)}</td>
        <td class="num">${x.r.timing_ms.total} ms</td></tr>`;
    }).join('') || '<tr><td colspan="7" class="muted">Sin datos todavía.</td></tr>';
  },
};

/* ============================================================
   Analizar imagen
   ============================================================ */
const Analyze = {
  last: null, // { kind: 'file'|'sample', file|id, img, res }
  filter: 'all',

  init() {
    const input = $('#fileInput'), drop = $('#drop');
    input.onchange = () => input.files[0] && this.runFile(input.files[0]);
    $('#camInput').onchange = (e) => { if (e.target.files[0]) this.runFile(e.target.files[0]); e.target.value = ''; };
    ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) this.runFile(f); });
    $('#conf').oninput = () => { $('#confVal').textContent = (+$('#conf').value).toFixed(2); };
    $('#conf').onchange = () => this.rerun();
    $('#mode').onchange = () => this.rerun();
    $('#showBoxes').onchange = $('#showMask').onchange = () => this.redraw();
    $$('#galFilter button').forEach((b) => b.onclick = () => {
      $$('#galFilter button').forEach((x) => x.classList.toggle('active', x === b));
      this.filter = b.dataset.f; this.renderGallery();
    });
    $('#galMore').onclick = () => this.renderGallery();
    window.addEventListener('resize', () => this.redraw());
  },

  renderGallery() {
    const g = $('#gallery');
    const list = App.samples.filter((s) => this.filter === 'all' || s.gt === +this.filter);
    if (!list.length) { g.innerHTML = '<p class="muted small">No hay imágenes de prueba en el servidor.</p>'; $('#galMore').hidden = true; return; }
    g.innerHTML = shuffle(list.slice()).slice(0, 20).map((s) => `
      <button data-id="${s.id}" title="${esc(s.file)} · real: ${CLS_ES[s.gt_label]}" style="background-image:url(/api/samples/${s.id}/image)">
        <span style="--c:${s.gt === 0 ? 'var(--bad)' : 'var(--ok)'}"></span></button>`).join('');
    $$('button', g).forEach((b) => b.onclick = () => { $$('button', g).forEach((x) => x.classList.toggle('sel', x === b)); this.runSample(+b.dataset.id); });
  },

  params() { return { conf: (+$('#conf').value).toFixed(2), mode: $('#mode').value }; },

  /* Las fotos del celular pesan varios MB: se reducen a 1600 px en el navegador antes de subirlas.
     El navegador ya aplica la orientación EXIF al dibujar, así que la imagen enviada sale derecha. */
  async prepare(file) {
    const url = URL.createObjectURL(file);
    const img = await loadImage(url);
    const k = Math.min(1, 1600 / Math.max(img.naturalWidth, img.naturalHeight));
    if (k === 1 && file.size < 2.5e6) return { blob: file, img, url };
    const c = document.createElement('canvas');
    c.width = Math.round(img.naturalWidth * k); c.height = Math.round(img.naturalHeight * k);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    const blob = await new Promise((r) => c.toBlob(r, 'image/jpeg', 0.9));
    return { blob, img, url };
  },

  async runFile(file) {
    const { conf, mode } = this.params();
    if (this.last?.url) URL.revokeObjectURL(this.last.url);
    $('#anaMeta').textContent = 'Subiendo…';
    let prep;
    try { prep = await this.prepare(file); } catch {
      $('#anaSummary').innerHTML = '<p class="bad-txt">El navegador no pudo abrir esa imagen (¿formato HEIC?). Prueba con JPG o PNG.</p>';
      $('#anaMeta').textContent = '';
      return;
    }
    const fd = new FormData(); fd.append('file', prep.blob, 'foto.jpg'); fd.append('conf', conf); fd.append('mode', mode);
    await this.show(api('/api/predict', { method: 'POST', body: fd }), Promise.resolve(prep.img), { kind: 'file', file, url: prep.url });
  },

  async runSample(id) {
    const { conf, mode } = this.params();
    await this.show(api(`/api/samples/${id}/predict?conf=${conf}&mode=${mode}`, { method: 'POST' }), loadImage(`/api/samples/${id}/image`), { kind: 'sample', id });
  },

  rerun() {
    if (!this.last) return;
    if (this.last.kind === 'file') this.runFile(this.last.file); else this.runSample(this.last.id);
  },

  async show(resP, imgP, src) {
    $('#anaMeta').textContent = 'Analizando…';
    try {
      const [res, img] = await Promise.all([resP.then(withMasks), imgP]);
      this.last = { ...src, img, res };
      $('#anaPlaceholder').hidden = true;
      this.redraw();
      this.renderDetails();
      if (window.innerWidth < 900) $('#anaCanvas').closest('.card').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      $('#anaMeta').textContent = '';
      $('#anaSummary').innerHTML = `<p class="bad-txt">No se pudo analizar: ${esc(err.message || err)}</p>`;
    }
  },

  redraw() {
    if (!this.last) return;
    drawResult($('#anaCanvas'), this.last.img, this.last.res, { maxH: 520, boxes: $('#showBoxes').checked, mask: $('#showMask').checked });
  },

  renderDetails() {
    const r = this.last.res;
    const gt = r.sample ? `<span>Etiqueta real: <b>${CLS_ES[r.sample.gt_label]}</b></span>` : '';
    $('#anaMeta').textContent = `${r.width}×${r.height} px · ${r.timing_ms.total} ms`;
    $('#anaSummary').innerHTML = `<div class="summary-line">${routePill(r.route, true)}
      <span><b>${r.summary.n_eggs}</b> huevo(s): ${r.summary.n_crack} rajado(s), ${r.summary.n_intact} sano(s)</span>${gt}
      <span class="muted">detector ${r.timing_ms.detector} ms · daño ${r.timing_ms.damage} ms</span></div>
      <p class="small muted">${esc(App.routes[r.route]?.desc || '')}</p>`;
    const box = $('#anaEggs');
    box.innerHTML = '';
    if (!r.eggs.length) box.innerHTML = '<p class="muted">El detector no encontró huevos con esa confianza. Prueba a bajar el umbral.</p>';
    r.eggs.forEach((e, i) => {
      const card = document.createElement('div');
      card.className = 'egg-card';
      card.innerHTML = `
        <h5>Huevo ${i + 1} <span class="route-pill" style="--c:${e.cls === 0 ? 'var(--bad)' : 'var(--ok)'}">${CLS_ES[e.label]} ${Math.round(e.conf * 100)} %</span></h5>
        <div class="scores">
          <span>Rajado</span><span class="track"><b style="width:${e.scores.Crack * 100}%;background:var(--bad)"></b></span><span class="num">${e.scores.Crack.toFixed(2)}</span>
          <span>Sano</span><span class="track"><b style="width:${e.scores.Intact * 100}%;background:var(--ok)"></b></span><span class="num">${e.scores.Intact.toFixed(2)}</span>
        </div>
        ${e.damage ? `<div class="small">Zona dañada: <b>${pct(e.damage.severity, 1)}</b> de la cáscara visible · gravedad <b>${LEVEL_ES[e.damage.level]}</b></div>${sevBar(e.damage.severity)}`
          : '<p class="small muted">Huevo sano: dano_v1 no se ejecuta.</p>'}`;
      if (e.damage) card.appendChild(cropPair(e));
      box.appendChild(card);
    });
    const slim = JSON.parse(JSON.stringify(r, (k, v) => (k.startsWith('_') ? undefined : typeof v === 'string' && v.startsWith('data:') ? `${v.slice(0, 40)}… (${v.length} caracteres)` : v)));
    $('#anaRaw').textContent = JSON.stringify(slim, null, 2);
  },
};

/* ============================================================
   Cámara / video en vivo
   ============================================================ */
const Live = {
  active: false, stream: null, res: null, hist: [], n: 0, t0: 0, latSum: 0, busy: false, url: null,

  init() {
    this.video = $('#liveVideo'); this.canvas = $('#liveCanvas');
    this.grab = document.createElement('canvas');
    $('#camStart').onclick = () => this.startCamera();
    $('#videoInput').onchange = (e) => { const f = e.target.files[0]; if (f) this.startVideo(f); e.target.value = ''; };
    $('#liveStop').onclick = () => this.stop();
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) $('#secureNotice').hidden = false;
  },

  async startCamera() {
    this.stop();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false });
    } catch (err) {
      $('#secureNotice').hidden = false;
      $('#liveStats').textContent = `No se pudo abrir la cámara: ${err.message || err}`;
      return;
    }
    this.video.srcObject = this.stream;
    await this.video.play();
    this.begin();
  },

  async startVideo(file) {
    this.stop();
    this.url = URL.createObjectURL(file);
    this.video.srcObject = null;
    this.video.src = this.url;
    await this.video.play();
    this.begin();
  },

  begin() {
    this.active = true; this.res = null; this.hist = []; this.n = 0; this.latSum = 0; this.t0 = performance.now();
    $('#livePlaceholder').hidden = true;
    const draw = () => {
      if (!this.active) return;
      if (this.video.readyState >= 2) drawResult(this.canvas, this.video, this.res, { maxH: 480, mask: $('#liveMask').checked });
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
    this.loop();
  },

  async loop() {
    while (this.active) {
      const t = performance.now();
      if (this.video.readyState >= 2 && this.video.videoWidth) {
        const v = this.video, k = Math.min(1, 800 / Math.max(v.videoWidth, v.videoHeight));
        this.grab.width = Math.round(v.videoWidth * k); this.grab.height = Math.round(v.videoHeight * k);
        this.grab.getContext('2d').drawImage(v, 0, 0, this.grab.width, this.grab.height);
        const blob = await new Promise((r) => this.grab.toBlob(r, 'image/jpeg', 0.85));
        const fd = new FormData(); fd.append('file', blob, 'frame.jpg'); fd.append('conf', '0.5');
        try {
          const res = await withMasks(await api('/api/predict', { method: 'POST', body: fd }));
          if (!this.active) break;
          this.res = res; this.n++; this.latSum += performance.now() - t;
          this.update(res);
        } catch (err) {
          $('#liveStats').textContent = `Error: ${err.message || err}`;
        }
      }
      const wait = 220 - (performance.now() - t); // máx. ~4-5 inferencias por segundo
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
  },

  update(res) {
    const main = res.eggs.slice().sort((a, b) => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]))[0];
    this.hist.push(main ? { cls: main.cls, sev: main.damage ? main.damage.severity : null } : { cls: null, sev: null });
    if (this.hist.length > 7) this.hist.shift();
    const votes = { 0: 0, 1: 0, none: 0 };
    this.hist.forEach((h) => { votes[h.cls == null ? 'none' : h.cls]++; });
    const cls = votes.none > votes[0] + votes[1] ? null : (votes[0] >= votes[1] ? 0 : 1);
    const sevs = this.hist.filter((h) => h.sev != null).map((h) => h.sev);
    const sev = sevs.length ? sevs.reduce((a, b) => a + b, 0) / sevs.length : 0;
    const route = cls == null ? 'revision' : cls === 1 ? 'empaque' : sev < 0.15 ? 'industria' : 'descarte';
    const secs = (performance.now() - this.t0) / 1000;
    $('#liveStats').textContent = `${(this.n / secs).toFixed(1)} inferencias/s · ${Math.round(this.latSum / this.n)} ms ida y vuelta · modelo ${res.timing_ms.total} ms`;
    $('#liveDecision').innerHTML = `
      <div class="big">${cls == null ? 'Sin huevo' : cls === 0 ? 'Rajado' : 'Sano'}</div>
      ${routePill(route, true)}
      <dl class="kv"><dt>Votos (7 fotogramas)</dt><dd>rajado ${votes[0]} · sano ${votes[1]} · sin huevo ${votes.none}</dd>
      <dt>Gravedad media</dt><dd>${cls === 0 ? pct(sev, 1) : '—'}</dd>
      <dt>Huevos en el fotograma</dt><dd>${res.eggs.length}</dd></dl>
      ${cls === 0 ? sevBar(sev) : ''}`;
  },

  stop() {
    this.active = false;
    if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
    if (this.video) { this.video.pause(); this.video.srcObject = null; }
    if (this.url) { URL.revokeObjectURL(this.url); this.url = null; }
  },
};

/* ============================================================
   Modelo y métricas
   ============================================================ */
function renderModelTab() {
  const i = App.info;
  const mb = (b) => `${(b / 1e6).toFixed(1)} MB`;
  const d = i.models.detector, g = i.models.damage;
  $('#modelCards').innerHTML = `
    <div class="card model-card"><h3>1 · Detector <code>v2</code> (obligatorio)</h3>
      <p class="small muted">YOLOv8n, fine-tune con ~2.600 imágenes sintéticas para que no aprenda el fondo. Encuentra cada huevo y lo clasifica.</p>
      <dl class="kv"><dt>Archivo</dt><dd><code>${esc(d.file)}</code> · ${mb(d.bytes)}</dd>
      <dt>Entrada</dt><dd>[${d.input}] float32 · RGB 0–1</dd><dt>Salida</dt><dd>[${d.output}] · cx, cy, w, h, score Crack, score Intact</dd>
      <dt>Clases</dt><dd>0 = Crack (rajado) · 1 = Intact (sano)</dd><dt>Decodificación</dt><dd>conf ≥ ${d.conf} · NMS agnóstica IoU ${d.iou_nms}</dd></dl></div>
    <div class="card model-card"><h3>2 · Zona dañada <code>dano_v1</code> (diferencial)</h3>
      <p class="small muted">U-Net con codificador MobileNetV2-0.5. Recibe el recorte de cada huevo rajado y devuelve dos máscaras.</p>
      <dl class="kv"><dt>Archivo</dt><dd><code>${esc(g.file)}</code> · ${mb(g.bytes)}</dd>
      <dt>Entrada</dt><dd>[${g.input}] float32 · caja +10 % estirada</dd><dt>Salida</dt><dd>[${g.output}] · silueta del huevo, zona dañada</dd>
      <dt>Gravedad</dt><dd>leve ${g.levels.leve} · media ${g.levels.media} · grave ${g.levels.grave}</dd>
      <dt>Anotaciones</dt><dd>420 huevos rajados marcados a mano (rejilla 10×10) + silueta con SAM 2.1</dd></dl></div>`;

  const v2 = i.results_v2;
  if (v2) {
    const t = v2.test_original_por_clase;
    const f = (x) => (x == null ? '—' : x.toFixed(3));
    const grp = Object.entries(v2.acierto_por_grupo || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${f(v)}</td></tr>`).join('');
    const ver = v2.verificacion || {};
    $('#metV2').innerHTML = `
      <p class="small muted">Test original: 382 fotos que el modelo no vio al entrenar.</p>
      <table><thead><tr><th>Clase</th><th class="num">Precisión</th><th class="num">Recall</th><th class="num">mAP50</th><th class="num">mAP50-95</th></tr></thead><tbody>
      ${['Crack', 'Intact'].map((c) => `<tr><td>${CLS_ES[c]}</td><td class="num">${f(t.P[c])}</td><td class="num">${f(t.R[c])}</td><td class="num">${f(t.mAP50[c])}</td><td class="num">${f(t['mAP50-95'][c])}</td></tr>`).join('')}
      </tbody></table>
      <h4>Acierto por grupo (conf 0.5)</h4><table><tbody>${grp}</tbody></table>
      <h4>.tflite frente al modelo original</h4>
      <table><thead><tr><th>Modelo</th><th class="num">mAP50</th><th class="num">mAP50-95</th></tr></thead><tbody>
      ${[['.pt original', ver.pt], ['fp32 .tflite (este)', ver.fp32], ['int8 .tflite', ver.int8]].filter(([, x]) => x).map(([n, x]) => `<tr><td>${n}</td><td class="num">${f(x.mAP50)}</td><td class="num">${f(x['mAP50-95'])}</td></tr>`).join('')}
      </tbody></table>`;
  } else $('#metV2').innerHTML = '<p class="muted">No se encontró resultados/v2/resumen.json.</p>';

  const dn = i.results_dano_v1;
  if (dn) {
    const t = dn.test?.tflite_fp16 || {}, p = dn.pipeline_test || {};
    $('#metDano').innerHTML = `
      <p class="small muted">Test: 35 huevos rajados y 30 sanos que el modelo no vio al entrenar.</p>
      <table><thead><tr><th>Métrica</th><th class="num">Recorte ideal</th><th class="num">Tubería v2 → dano_v1</th></tr></thead><tbody>
        <tr><td>IoU de la silueta del huevo</td><td class="num">${t.egg_iou?.toFixed(3) ?? '—'}</td><td class="num">—</td></tr>
        <tr><td>IoU de la zona dañada (rajados)</td><td class="num">${t.dmg_iou_pos?.toFixed(3) ?? '—'}</td><td class="num">${p.dmg_iou_pos?.toFixed(3) ?? '—'}</td></tr>
        <tr><td>Error medio de la gravedad</td><td class="num">±${((t.sev_mae_pos || 0) * 100).toFixed(1)} pts</td><td class="num">±${((p.sev_mae_pos || 0) * 100).toFixed(1)} pts</td></tr>
        <tr><td>Rajados con daño detectado (&gt; 3 %)</td><td class="num">${pct(t.pos_con_dano)}</td><td class="num">${pct(p.pos_con_dano)}</td></tr>
        <tr><td>Sanos con daño detectado (&gt; 3 %)</td><td class="num">${pct(t.neg_con_dano)}</td><td class="num">${pct(p.neg_con_dano)}</td></tr>
        <tr><td>IoU medio de la caja de v2</td><td class="num">—</td><td class="num">${p.box_iou_medio?.toFixed(3) ?? '—'}</td></tr>
      </tbody></table>
      <p class="small muted">FP16 (el que usa esta app) da las mismas métricas que FP32 con la mitad de tamaño. La gravedad predicha suele quedar algo por debajo de la anotada, porque las zonas se marcaron sobre una rejilla gruesa.</p>`;
  } else $('#metDano').innerHTML = '<p class="muted">No se encontró resultados/dano_v1/resumen.json.</p>';

  const ev = i.server_eval;
  if (ev) {
    const c = ev.confusion;
    $('#metServer').innerHTML = `
      <p class="small muted">Esta misma app (<code>evaluate.py</code>) recorrió las ${ev.n_imagenes} imágenes del split <b>${esc(ev.split)}</b> en esta instancia de AWS, con conf ${ev.conf} (${esc(ev.fecha)}). Reproduce las cifras del equipo, lo que confirma que el preprocesado y la decodificación del servidor son correctos.</p>
      <div class="grid-2">
        <div><table><thead><tr><th>Grupo</th><th class="num">n</th><th class="num">Acierto</th></tr></thead><tbody>
          ${ev.grupos.map((g) => `<tr><td>${esc(g.grupo.replace('Crack', 'Rajado').replace('Intact', 'Sano'))}</td><td class="num">${g.n}</td><td class="num">${g.acierto == null ? '—' : g.acierto.toFixed(3)}</td></tr>`).join('')}
          <tr><th>Global</th><th class="num">${ev.n_imagenes}</th><th class="num">${ev.acierto_global.toFixed(3)}</th></tr></tbody></table></div>
        <div><table class="confusion"><thead><tr><th>Real \\ Modelo</th><th>Rajado</th><th>Sano</th><th>Ninguno</th></tr></thead><tbody>
          <tr><th>Rajado</th><td class="hit">${c.Crack.Crack}</td><td class="${c.Crack.Intact ? 'miss' : ''}">${c.Crack.Intact}</td><td class="${c.Crack.Ninguno ? 'miss' : ''}">${c.Crack.Ninguno}</td></tr>
          <tr><th>Sano</th><td class="${c.Intact.Crack ? 'miss' : ''}">${c.Intact.Crack}</td><td class="hit">${c.Intact.Intact}</td><td class="${c.Intact.Ninguno ? 'miss' : ''}">${c.Intact.Ninguno}</td></tr></tbody></table>
          <dl class="kv"><dt>Latencia (CPU de la instancia)</dt><dd>p50 ${ev.latencia_ms.p50} ms · p90 ${ev.latencia_ms.p90} ms</dd>
          <dt>Rutas asignadas</dt><dd>${ROUTE_KEYS.filter((k) => ev.rutas[k]).map((k) => `${esc(App.routes[k].name)}: ${ev.rutas[k]}`).join(' · ')}</dd>
          <dt>Gravedad media (rajados)</dt><dd>${pct(ev.gravedad_media_crack, 1)}</dd></dl></div>
      </div>`;
  } else $('#metServer').innerHTML = '<p class="muted">Aún no se ha ejecutado <code>evaluate.py</code> en el servidor.</p>';

  $('#routesTable').innerHTML = `<table><thead><tr><th>Ruta</th><th>Regla</th></tr></thead><tbody>
    ${ROUTE_KEYS.map((k) => `<tr><td>${routePill(k)}</td><td><b>${esc(App.routes[k].rule)}</b><br><span class="small muted">${esc(App.routes[k].desc)}</span></td></tr>`).join('')}
    </tbody></table>
    <p class="small muted">Con un detector binario, todo huevo rajado se descarta. La zona dañada permite aprovechar los que tienen una grieta pequeña.</p>`;
}

/* ============================================================
   Arranque
   ============================================================ */
async function boot() {
  readColors();
  $$('.host').forEach((e) => { e.textContent = location.hostname; });
  $$('.origin').forEach((e) => { e.textContent = location.origin; });
  initTabs();
  try {
    const [info, samples] = await Promise.all([api('/api/info'), api('/api/samples')]);
    App.info = info; App.samples = samples; App.routes = info.routes;
    $('#status').classList.add('ok');
    $('#statusText').textContent = `Modelos cargados · ${samples.length} imágenes de test`;
  } catch (err) {
    $('#status').classList.add('err');
    $('#statusText').textContent = 'Sin conexión con la API';
    console.error(err);
    return;
  }
  Belt.init();
  Analyze.init();
  Analyze.renderGallery();
  Live.init();
  renderModelTab();
}

boot();
