# Exporta el modelo de zona dañada a TFLite, verifica equivalencia y evalúa la tubería completa (detector v2 -> recorte -> segmentador)
import os, sys, json, glob, cv2, numpy as np, tensorflow as tf
from segcommon import S, PAD, load_split, crop

RUN = sys.argv[1]; OUT = sys.argv[2]; DET = sys.argv[3]
os.makedirs(OUT, exist_ok=False)
RUNS = os.environ.get('RUNS_DIR', 'runs')
model = tf.keras.models.load_model(f'{RUNS}/{RUN}/best.keras', compile=False)

# ---------- exportación ----------
conv = tf.lite.TFLiteConverter.from_keras_model(model)
fp32 = conv.convert(); open(f'{OUT}/eggs_{RUN}_fp32.tflite', 'wb').write(fp32)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]; conv.target_spec.supported_types = [tf.float16]
fp16 = conv.convert(); open(f'{OUT}/eggs_{RUN}_fp16.tflite', 'wb').write(fp16)

def interp(buf):
    it = tf.lite.Interpreter(model_content=buf); it.allocate_tensors(); return it
def run(it, x):
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i['index'], x); it.invoke(); return it.get_tensor(o['index'])

I32, I16 = interp(fp32), interp(fp16)
io = dict(input=[(d['name'], d['shape'].tolist(), str(d['dtype'].__name__)) for d in I32.get_input_details()],
          output=[(d['name'], d['shape'].tolist(), str(d['dtype'].__name__)) for d in I32.get_output_details()])

# ---------- métricas sobre recortes de la etiqueta (test) ----------
def metrics(P, Y, pos):
    P, Y = P > .5, Y > .5
    iou = lambda a, b: (a & b).sum((1, 2)) / np.maximum(1, (a | b).sum((1, 2)))
    fp = P[..., 1].sum((1, 2)) / np.maximum(1, P[..., 0].sum((1, 2)))
    fy = Y[..., 1].sum((1, 2)) / np.maximum(1, Y[..., 0].sum((1, 2)))
    return dict(n_pos=int(pos.sum()), n_neg=int((~pos).sum()),
                egg_iou=round(float(iou(P[..., 0], Y[..., 0]).mean()), 4),
                dmg_iou_pos=round(float(iou(P[..., 1], Y[..., 1])[pos].mean()), 4),
                sev_mae_pos=round(float(np.abs(fp - fy)[pos].mean()), 4),
                pos_con_dano=round(float((fp[pos] > .03).mean()), 4),
                neg_con_dano=round(float((fp[~pos] > .03).mean()), 4) if (~pos).any() else None)

res = {'io': io, 'bytes': {'fp32': len(fp32), 'fp16': len(fp16)}}
for split in ['valid', 'test']:
    items = load_split(split)
    X = np.stack([crop(it)[0] for it in items]).astype(np.float32) / 255.
    Y = np.stack([crop(it)[1] for it in items]); pos = np.array([it['pos'] for it in items])
    K = model.predict(X, verbose=0)
    T32 = np.concatenate([run(I32, X[i:i + 1]) for i in range(len(X))])
    T16 = np.concatenate([run(I16, X[i:i + 1]) for i in range(len(X))])
    res[split] = {'keras': metrics(K, Y, pos), 'tflite_fp32': metrics(T32, Y, pos), 'tflite_fp16': metrics(T16, Y, pos),
                  'maxdiff_fp32': float(np.abs(K - T32).max()), 'maxdiff_fp16': float(np.abs(K - T16).max())}

# ---------- tubería completa en test: cajas del detector v2 ----------
D = tf.lite.Interpreter(model_path=DET); D.allocate_tensors()
def detect(img):  # img RGB completo; recorte cuadrado centrado como en la app
    h, w = img.shape[:2]; side = min(h, w); ox, oy = (w - side) // 2, (h - side) // 2
    x = cv2.resize(img[oy:oy + side, ox:ox + side], (640, 640)).astype(np.float32)[None] / 255.
    o = run(D, x)[0]; conf = o[4:].max(0); cls = o[4:].argmax(0); keep = np.where(conf >= .5)[0]
    keep = keep[np.argsort(-conf[keep])]; boxes = []
    for i in keep:
        cx, cy, bw, bh = o[:4, i]; b = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        if all(iou1(b, c[0]) < .5 for c in boxes): boxes.append((b, int(cls[i]), float(conf[i])))
    return [(b * side + [ox, oy, ox, oy], c, s) for b, c, s in boxes]
def iou1(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0

items = load_split('test'); pipe = []; vis = []
for it in items:
    dets = detect(it['img'])
    if not dets: pipe.append(dict(name=os.path.basename(it['name']), found=False)); continue
    j = int(np.argmax([iou1(d[0], it['box']) for d in dets])); b, c, s = dets[j]
    it2 = dict(it, box=np.float32(b)); x, y = crop(it2)
    p = run(I32, x[None].astype(np.float32) / 255.)[0] > .5; y = y > .5
    fp = p[..., 1].sum() / max(1, p[..., 0].sum()); fy = y[..., 1].sum() / max(1, y[..., 0].sum())
    di = (p[..., 1] & y[..., 1]).sum() / max(1, (p[..., 1] | y[..., 1]).sum())
    pipe.append(dict(name=os.path.basename(it['name']), found=True, box_iou=round(iou1(b, it['box']), 3), cls=c,
                     pos=bool(it['pos']), dmg_iou=round(float(di), 3), sev_pred=round(float(fp), 3), sev_gt=round(float(fy), 3)))
    if len(vis) < 12 and it['pos']: vis.append((x, p))
P = [r for r in pipe if r['found']]
res['pipeline_test'] = dict(
    n=len(pipe), detectados=len(P), box_iou_medio=round(float(np.mean([r['box_iou'] for r in P])), 4),
    dmg_iou_pos=round(float(np.mean([r['dmg_iou'] for r in P if r['pos']])), 4),
    sev_mae_pos=round(float(np.mean([abs(r['sev_pred'] - r['sev_gt']) for r in P if r['pos']])), 4),
    pos_con_dano=round(float(np.mean([r['sev_pred'] > .03 for r in P if r['pos']])), 4),
    neg_con_dano=round(float(np.mean([r['sev_pred'] > .03 for r in P if not r['pos']])), 4))
res['pipeline_detalle'] = pipe
json.dump(res, open(f'{OUT}/resumen.json', 'w'), indent=1)

# ---------- ejemplos visuales ----------
T = []
for x, p in vis:
    v = x.copy(); ov = v.copy(); ov[p[..., 1]] = (255, 0, 255); v = cv2.addWeighted(v, .55, ov, .45, 0)
    cn, _ = cv2.findContours(p[..., 0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cn, -1, (0, 255, 0), 1); T.append(np.hstack([x, v]))
cv2.imwrite(f'{OUT}/ejemplos_test.jpg', cv2.cvtColor(np.vstack([np.hstack(T[i:i + 3]) for i in range(0, len(T) - len(T) % 3, 3)]), cv2.COLOR_RGB2BGR))
print(json.dumps({k: v for k, v in res.items() if k != 'pipeline_detalle'}, indent=1))
