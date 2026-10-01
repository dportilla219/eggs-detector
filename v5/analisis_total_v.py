"""Solo análisis: v4 sobre TODAS las imágenes de huevos disponibles, separando las que vio al entrenar
de las que no. Guarda los puntajes por huevo para comparar umbrales de confianza."""
import glob, json, os, sys
import numpy as np
from PIL import Image

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(R, "repo", "app", "server"))
from pipeline import EggPipeline, to_rgb  # noqa: E402

DET = sys.argv[1] if len(sys.argv) > 1 else "eggs_v4_fp32.tflite"
SAL = sys.argv[2] if len(sys.argv) > 2 else "analisis_total.json"
p = EggPipeline(os.path.join(R, "repo", "modelo"), det_file=DET)
items = []  # (grupo, visto_al_entrenar, clase, ruta)
for split in ("train", "valid", "test"):
    D = os.path.join(R, "data", "eggs_v2", split)
    for f in sorted(glob.glob(os.path.join(D, "images", "*"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        if "_dup" in stem: continue
        cl = [int(l.split()[0]) for l in open(os.path.join(D, "labels", stem + ".txt")) if l.strip()]
        if not cl: continue
        items.append((f"original {'montaje' if stem.startswith('ec_egg') else 'otras fuentes'}", split == "train", "Crack" if 0 in cl else "Intact", f))
aud = {e["archivo"]: e for e in json.load(open(os.path.join(R, "resultados_v4", "auditoria", "etiquetas.json"), encoding="utf-8"))}
for carpeta, clase in (("non defective", "Intact"), ("defective", "Crack")):
    for f in sorted(glob.glob(os.path.join(R, "public", carpeta, "*"))):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
        e = aud.get(os.path.basename(f))
        items.append(("publico (celular)", bool(e and e.get("split") == "train"), clase, f))
trn = {x[5] for x in json.load(open(os.path.join(R, "commons", "entrenamiento_commons.json"), encoding="utf-8"))}
for url, clase, *_r, i in json.load(open(os.path.join(R, "commons", "prueba_commons.json"), encoding="utf-8")):
    items.append(("commons", False, clase, os.path.join(R, "commons", "cand", f"{i:04d}.jpg")))
for i in trn:
    items.append(("commons", True, "Intact", os.path.join(R, "commons", "cand", f"{i:04d}.jpg")))
lista = {x["id"]: x for x in json.load(open(os.path.join(R, "commons", "aleatorias", "lista.json"), encoding="utf-8"))}
for i in (5, 14, 18, 26, 29, 62): items.append(("nuevas", False, "Intact", os.path.join(R, "commons", "aleatorias", lista[i]["file"])))
for i in (224, 392, 299, 174, 225, 538, 316, 506, 170, 80, 395, 507, 77, 302, 227, 138, 190, 123, 389, 96, 279, 201, 189, 143, 542):
    items.append(("nuevas", False, "Intact", os.path.join(R, "commons", "cand", f"{i:04d}.jpg")))
for i in (24, 25, 55): items.append(("nuevas", False, "Crack", os.path.join(R, "commons", "aleatorias", lista[i]["file"])))
for i in (6, 41, 9, 17, 5, 4, 7): items.append(("nuevas", False, "Crack", os.path.join(R, "commons", "cand", f"{i:04d}.jpg")))
print(len(items), "imágenes", flush=True)

filas = []
for k, (g, visto, gt, f) in enumerate(items):
    try:
        img = to_rgb(Image.open(f))
    except Exception:
        continue
    if max(img.size) > 2048: img.thumbnail((2048, 2048))
    x, lb, _ = p._det_input(img, "letterbox")
    p.det.set_tensor(p._det_in, x); p.det.invoke()
    h = [[round(float(sc), 4), round(float(si), 4)] for box, c, cls, sc, si in p._decode(p.det.get_tensor(p._det_out)[0], 0.2)]
    filas.append({"g": g, "visto": visto, "gt": gt, "h": h, "f": os.path.basename(f)})
    if k % 500 == 0: print(k, flush=True)
json.dump(filas, open(os.path.join(R, SAL), "w"))


def pred(h, conf):
    hs = [e for e in h if max(e) >= conf]
    return "Ninguno" if not hs else ("Crack" if any(sc >= si for sc, si in hs) else "Intact")


for conf in (0.5, 0.4, 0.3):
    print(f"\n=== umbral de confianza {conf} ===")
    for visto in (False, True):
        print("  NO vistas al entrenar:" if not visto else "  vistas al entrenar:")
        tot = {"Crack": [0, 0], "Intact": [0, 0]}
        for g in sorted({f["g"] for f in filas}):
            for k in ("Crack", "Intact"):
                sub = [f for f in filas if f["g"] == g and f["visto"] == visto and f["gt"] == k]
                if not sub: continue
                ok = sum(pred(f["h"], conf) == k for f in sub); nin = sum(pred(f["h"], conf) == "Ninguno" for f in sub)
                tot[k][0] += ok; tot[k][1] += len(sub)
                print(f"    {g:24s} {k:6s} {ok}/{len(sub)} = {ok / len(sub):.3f}  (sin detección: {nin})")
        for k, (a, b) in tot.items():
            if b: print(f"    TOTAL {k}: {a}/{b} = {a / b:.3f}")
