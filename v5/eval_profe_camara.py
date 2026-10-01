"""Imágenes del profesor como si se les apuntara con la cámara: perspectiva, borde, brillo, desenfoque y JPEG al azar.
Acierto por huevo con la tubería de la app (varias repeticiones por imagen, semillas distintas de las del entrenamiento)."""
import json, os, sys
import numpy as np, cv2
from PIL import Image
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(R, "repo", "app", "server")); sys.path.insert(0, os.path.join(R, "repo", "v5"))
from pipeline import EggPipeline  # noqa: E402
import datos  # noqa: E402
DET = sys.argv[1]; REP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
p = EggPipeline(os.path.join(R, "repo", "modelo"), det_file=DET)
H = json.load(open(os.path.join(R, "repo", "v5", "profe_huevos.json"), encoding="utf-8"))
rng = np.random.default_rng(777)
def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0
tot = {"suelta": [0, 0, 0], "lamina": [0, 0, 0]}; sobran = 0
for f, d in H.items():
    img = datos.leer_rgb(os.path.join(R, "repo_profe", "imagenes", f))
    cajas = [(e["clase"], *e["caja"]) for e in d["huevos"] if not e["omitir"]]
    k = "lamina" if f.startswith("plantilla") else "suelta"
    for _ in range(REP):
        im2, c2 = datos.pantalla(img, cajas, rng)
        if rng.random() < 0.5: im2 = cv2.GaussianBlur(im2, (0, 0), rng.uniform(0.6, 1.4))
        ok, b = cv2.imencode(".jpg", cv2.cvtColor(im2, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(40, 85))])
        im2 = cv2.cvtColor(cv2.imdecode(b, 1), cv2.COLOR_BGR2RGB)
        r = p.predict(Image.fromarray(im2), conf=0.5, with_damage=False)
        us = set()
        for (cl, *bx) in c2:
            m = [(iou(bx, x["box"]), j) for j, x in enumerate(r["eggs"])]; m = [x for x in m if x[0] >= 0.45]
            tot[k][1] += 1
            if not m: tot[k][2] += 1; continue
            j = max(m)[1]; us.add(j); tot[k][0] += r["eggs"][j]["cls"] == cl
        sobran += len(r["eggs"]) - len(us)
for k, (a, b, c) in tot.items(): print(f"{DET} {k}: {a}/{b} = {a / b:.3f} ({c} sin detección)")
print("detecciones de más:", sobran)
