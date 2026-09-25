'use strict';
/* ============================================================
   Motor local: los mismos modelos .tflite, ejecutados en el navegador con TFLite WebAssembly.
   Responde a las mismas rutas /api/* que el servidor (server/main.py) con la misma tubería
   (server/pipeline.py + server/routing.py). Lo usa la versión sin servidor de la app, que se
   activa definiendo window.EGGS_LOCAL antes de cargar app.js:
     { modelos: 'modelo/', det: 'eggs_v2_fp32.tflite', seg: 'eggs_dano_v1_fp16.tflite',
       wasm: 'wasm/', datos: 'datos/' }
   det/seg también pueden ser { archivo, partes: ['x.b64.1.txt', ...] } (el .tflite en base64 por trozos).
   Requiere tfjs-core, tfjs-backend-cpu y tfjs-tflite cargados antes.
   ============================================================ */
const Local = (() => {
  const DET = 640, SEG = 192, IOU = 0.5, PAD = 0.05, MAX_SIDE = 2048, MIN_SIDE = 32;
  const LEVELS = [[0.15, 'leve'], [0.35, 'media'], [1.01, 'grave']];
  const LEVE_MAX = 0.15;
  const DMG_RGB = [236, 64, 200];
  const CLASSES = ['Crack', 'Intact'];
  let cfg, det, seg, info, samples;

  const level = (s) => (LEVELS.find(([lim]) => s < lim) || [0, 'grave'])[1];
  const r1 = (v) => Math.round(v * 10) / 10;
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  const ctx2d = (c) => c.getContext('2d', { willReadFrequently: true });

  const pedir = async (url, tipo) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r[tipo]();
  };

  /* Un modelo es un .tflite o, donde no se pueden servir binarios, una lista de trozos de texto base64. */
  async function modelo(m) {
    if (typeof m === 'string') return tflite.loadTFLiteModel(cfg.modelos + m, { numThreads: 1 });
    const b64 = (await Promise.all(m.partes.map((p) => pedir(cfg.modelos + p, 'text')))).join('').replace(/\s+/g, '');
    const bin = atob(b64), buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    return tflite.loadTFLiteModel(buf.buffer, { numThreads: 1 });
  }
  const nombre = (m) => (typeof m === 'string' ? m : m.archivo);

  async function init(c) {
    cfg = c;
    tflite.setWasmPath(new URL(cfg.wasm, location.href).href);
    [det, seg, info, samples] = await Promise.all([
      modelo(cfg.det), modelo(cfg.seg), pedir(cfg.datos + 'info.json', 'json'), pedir(cfg.datos + 'samples.json', 'json'),
    ]);
    // comprobación de entrada/salida, igual que EggPipeline._check_io
    const shp = (m, k) => JSON.stringify(m[k][0].shape);
    if (shp(det, 'inputs') !== '[1,640,640,3]' || shp(det, 'outputs') !== '[1,6,8400]'
        || shp(seg, 'inputs') !== '[1,192,192,3]' || shp(seg, 'outputs') !== '[1,192,192,2]') {
      throw new Error('Los modelos no tienen la entrada/salida esperada.');
    }
    run(det, new Float32Array(DET * DET * 3), [1, DET, DET, 3]); // calentamiento
    run(seg, new Float32Array(SEG * SEG * 3), [1, SEG, SEG, 3]);
  }

  function run(model, data, shape) {
    const x = tf.tensor(data, shape, 'float32');
    const y = model.predict(x);
    const out = y.dataSync().slice();
    x.dispose(); y.dispose();
    return out;
  }

  /* RGBA de un canvas -> Float32 RGB 0..1 (NHWC) */
  function toInput(c, S) {
    const px = ctx2d(c).getImageData(0, 0, S, S).data;
    const a = new Float32Array(S * S * 3);
    for (let i = 0, j = 0; i < px.length; i += 4, j += 3) { a[j] = px[i] / 255; a[j + 1] = px[i + 1] / 255; a[j + 2] = px[i + 2] / 255; }
    return a;
  }

  /* Imagen/vídeo/canvas -> canvas RGB con fondo blanco, como máximo MAX_SIDE px (igual que main._run) */
  function frame(src) {
    const w = src.naturalWidth || src.videoWidth || src.width, h = src.naturalHeight || src.videoHeight || src.height;
    if (!w || !h) throw new Error('No se pudo leer la imagen.');
    if (Math.min(w, h) < MIN_SIDE) throw new Error(`La imagen es demasiado pequeña (${w}×${h} px; mínimo ${MIN_SIDE} px por lado).`);
    const k = Math.min(1, MAX_SIDE / Math.max(w, h));
    const c = canvas(Math.round(w * k), Math.round(h * k));
    const g = ctx2d(c);
    g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height); // transparencia sobre blanco
    g.imageSmoothingQuality = 'high';
    g.drawImage(src, 0, 0, c.width, c.height);
    return c;
  }

  function detInput(img, mode) {
    const W = img.width, H = img.height, c = canvas(DET, DET), g = ctx2d(c);
    g.imageSmoothingQuality = 'high';
    if (mode === 'center') {
      const side = Math.min(W, H), ox = Math.floor((W - side) / 2), oy = Math.floor((H - side) / 2);
      g.drawImage(img, ox, oy, side, side, 0, 0, DET, DET);
      return { x: toInput(c, DET), lb: { scale: DET / side, px: -ox * DET / side, py: -oy * DET / side }, square: [ox, oy, side] };
    }
    const scale = DET / Math.max(W, H);
    const nw = Math.max(1, Math.round(W * scale)), nh = Math.max(1, Math.round(H * scale));
    const px = Math.floor((DET - nw) / 2), py = Math.floor((DET - nh) / 2);
    g.fillStyle = 'rgb(114,114,114)'; g.fillRect(0, 0, DET, DET);
    g.drawImage(img, px, py, nw, nh);
    return { x: toInput(c, DET), lb: { scale, px, py }, square: [0, 0, Math.max(W, H)] };
  }

  function iou(a, b) {
    const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = iw * ih, union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return union > 0 ? inter / union : 0;
  }

  /* [6,8400] -> cajas (0..640) tras NMS agnóstica, como EggPipeline._decode */
  function decode(out, thr) {
    const N = 8400, cand = [];
    for (let i = 0; i < N; i++) {
      const sc = out[4 * N + i], si = out[5 * N + i], conf = Math.max(sc, si);
      if (conf < thr) continue;
      const cx = out[i] * DET, cy = out[N + i] * DET, bw = out[2 * N + i] * DET, bh = out[3 * N + i] * DET;
      cand.push({ box: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], conf, sc, si });
    }
    cand.sort((a, b) => b.conf - a.conf);
    const chosen = [];
    for (const c of cand) if (chosen.every((k) => iou(c.box, k.box) < IOU)) chosen.push(c);
    return chosen.map((c) => ({ ...c, cls: c.sc >= c.si ? 0 : 1 }));
  }

  /* Máscara RGBA del daño, reescalada al tamaño del recorte (máx. 384 px), como EggPipeline._mask_png */
  function maskPng(prob, rw, rh) {
    const small = canvas(SEG, SEG), sg = ctx2d(small), id = sg.createImageData(SEG, SEG);
    for (let i = 0; i < SEG * SEG; i++) { const v = Math.round(Math.min(1, Math.max(0, prob[i])) * 255); id.data.set([v, v, v, 255], i * 4); }
    sg.putImageData(id, 0, 0);
    const s = Math.min(1, 384 / Math.max(rw, rh, 1));
    const tw = Math.max(8, Math.round(rw * s)), th = Math.max(8, Math.round(rh * s));
    const big = canvas(tw, th), bg = ctx2d(big);
    bg.imageSmoothingEnabled = true;
    bg.drawImage(small, 0, 0, tw, th);
    const p = bg.getImageData(0, 0, tw, th).data;
    const m = new Uint8Array(tw * th);
    for (let i = 0; i < m.length; i++) m[i] = p[i * 4] > 127 ? 1 : 0;
    const out = bg.createImageData(tw, th);
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const i = y * tw + x;
        if (!m[i]) continue;
        const inner = (y === 0 || m[i - tw]) && (y === th - 1 || m[i + tw]) && (x === 0 || m[i - 1]) && (x === tw - 1 || m[i + 1]);
        out.data.set([...DMG_RGB, inner ? 120 : 255], i * 4);
      }
    }
    bg.clearRect(0, 0, tw, th);
    bg.putImageData(out, 0, 0);
    return big.toDataURL('image/png');
  }

  /* Recorte +5 % por lado estirado a 192×192 -> dano_v1, como EggPipeline._segment */
  function segment(img, box) {
    const W = img.width, H = img.height, [x1, y1, x2, y2] = box, bw = x2 - x1, bh = y2 - y1;
    const rx = Math.max(0, x1 - PAD * bw), ry = Math.max(0, y1 - PAD * bh);
    const rx2 = Math.min(W, x2 + PAD * bw), ry2 = Math.min(H, y2 + PAD * bh), rw = rx2 - rx, rh = ry2 - ry;
    const c = canvas(SEG, SEG), g = ctx2d(c);
    g.imageSmoothingQuality = 'high';
    g.drawImage(img, rx, ry, rw, rh, 0, 0, SEG, SEG);
    const out = run(seg, toInput(c, SEG), [1, SEG, SEG, 3]);
    const prob = new Float32Array(SEG * SEG);
    let eggPx = 0, dmgPx = 0;
    for (let i = 0; i < SEG * SEG; i++) {
      const e = out[2 * i] > 0.5, d = out[2 * i + 1];
      if (e) { eggPx++; if (d > 0.5) dmgPx++; prob[i] = d; }
    }
    const sev = eggPx ? dmgPx / eggPx : 0;
    return { rect: [r1(rx), r1(ry), r1(rw), r1(rh)], severity: r4(sev), level: level(sev), egg_px: eggPx, damage_px: dmgPx,
      mask_png: maskPng(prob, rw, rh), crop_png: c.toDataURL('image/jpeg', 0.85) };
  }

  function route(s) {
    if (s.n_eggs === 0) return 'revision';
    if (s.n_crack === 0) return 'empaque';
    return (s.max_severity || 0) < LEVE_MAX ? 'industria' : 'descarte';
  }

  function predict(src, { conf = 0.5, mode = 'letterbox', withDamage = true } = {}) {
    if (mode !== 'letterbox' && mode !== 'center') throw new Error("mode debe ser 'letterbox' o 'center'");
    if (!(conf >= 0.05 && conf <= 0.95)) throw new Error('conf debe estar entre 0.05 y 0.95');
    const img = frame(src), W = img.width, H = img.height;
    const t0 = performance.now();
    const { x, lb, square } = detInput(img, mode);
    const raw = run(det, x, [1, DET, DET, 3]);
    const t1 = performance.now();
    const eggs = [];
    for (const d of decode(raw, conf)) {
      const bx = [(d.box[0] - lb.px) / lb.scale, (d.box[1] - lb.py) / lb.scale, (d.box[2] - lb.px) / lb.scale, (d.box[3] - lb.py) / lb.scale];
      bx[0] = Math.min(Math.max(bx[0], 0), W); bx[2] = Math.min(Math.max(bx[2], 0), W);
      bx[1] = Math.min(Math.max(bx[1], 0), H); bx[3] = Math.min(Math.max(bx[3], 0), H);
      if (bx[2] - bx[0] < 2 || bx[3] - bx[1] < 2) continue;
      eggs.push({ box: bx.map(r1), cls: d.cls, label: CLASSES[d.cls], conf: r4(d.conf), scores: { Crack: r4(d.sc), Intact: r4(d.si) },
        damage: d.cls === 0 && withDamage ? segment(img, bx) : null });
    }
    const t2 = performance.now();
    const nCrack = eggs.filter((e) => e.cls === 0).length;
    const sevs = eggs.filter((e) => e.damage).map((e) => e.damage.severity);
    const worst = sevs.length ? Math.max(...sevs) : null;
    const summary = { n_eggs: eggs.length, n_crack: nCrack, n_intact: eggs.length - nCrack, max_severity: worst, max_level: worst == null ? null : level(worst) };
    return { width: W, height: H, mode, square, eggs, summary, route: route(summary), engine: 'navegador',
      timing_ms: { detector: r1(t1 - t0), damage: r1(t2 - t1), total: r1(t2 - t0) } };
  }

  const imgFromUrl = (url) => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error('No se pudo abrir la imagen.')); im.src = url; });

  async function imgFromBlob(blob) {
    if (!blob || !blob.size) throw new Error('El archivo está vacío.');
    if (blob.size > 15 * 1024 * 1024) throw new Error('La imagen supera 15 MB.');
    const url = URL.createObjectURL(blob);
    try { return await imgFromUrl(url); } catch {
      throw new Error('Este navegador no puede abrir ese archivo. Usa JPG, PNG o WEBP (las fotos HEIC de iPhone solo se pueden en la versión con servidor).');
    } finally { URL.revokeObjectURL(url); }
  }

  const sampleUrl = (id) => `${cfg.datos}muestras/${encodeURIComponent(samples[id].file)}`;
  const sample = (id) => { if (!(id >= 0 && id < samples.length)) throw new Error('Muestra no encontrada'); return samples[id]; };
  const nextFrame = () => new Promise((r) => setTimeout(r, 0)); // deja pintar la interfaz antes de inferir

  /* Mismo contrato que api() de app.js: devuelve el JSON que devolvería el servidor o lanza Error. */
  async function handle(path, opts = {}) {
    const [p, q = ''] = path.split('?');
    const qs = new URLSearchParams(q);
    if (p === '/api/health') return { status: 'ok', models: [nombre(cfg.det), nombre(cfg.seg)], samples: samples.length, engine: 'navegador' };
    if (p === '/api/info') return info;
    if (p === '/api/samples') return samples;
    let m = p.match(/^\/api\/samples\/(\d+)\/predict$/);
    if (m) {
      const s = sample(+m[1]);
      const img = await imgFromUrl(sampleUrl(s.id));
      await nextFrame();
      return { ...predict(img, { conf: +(qs.get('conf') || 0.5), mode: qs.get('mode') || 'letterbox' }), sample: s };
    }
    if (p === '/api/predict') {
      const fd = opts.body;
      const img = await imgFromBlob(fd.get('file'));
      await nextFrame();
      return predict(img, { conf: +(fd.get('conf') || 0.5), mode: fd.get('mode') || 'letterbox' });
    }
    throw new Error(`Ruta no disponible sin servidor: ${p}`);
  }

  return { init, handle, predict, sampleUrl };
})();
