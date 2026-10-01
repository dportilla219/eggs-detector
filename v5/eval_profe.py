"""Acierto por huevo en las imágenes del profesor (cajas de v5/profe_huevos.json) con la tubería de la app."""
import json, os, sys
import numpy as np
from PIL import Image
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(R, "repo", "app", "server"))
from pipeline import EggPipeline, to_rgb  # noqa: E402
DET = sys.argv[1]; CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
p = EggPipeline(os.path.join(R, "repo", "modelo"), det_file=DET)
H = json.load(open(os.path.join(R, "repo", "v5", "profe_huevos.json"), encoding="utf-8"))
def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0
tot = {"suelta": [0, 0, 0], "lamina": [0, 0, 0]}; fallos = []; sobran = 0
for f, d in H.items():
    im = Image.open(os.path.join(R, "repo_profe", "imagenes", f)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    r = p.predict(bg.convert("RGB"), conf=CONF, with_damage=False)
    k = "lamina" if f.startswith("plantilla") else "suelta"
    usados = set()
    for e in d["huevos"]:
        if e["omitir"]: continue
        m = [(iou(e["caja"], x["box"]), j) for j, x in enumerate(r["eggs"])]
        m = [x for x in m if x[0] >= 0.5]
        tot[k][1] += 1
        if not m: tot[k][2] += 1; fallos.append((e["id"], "sin detección")); continue
        j = max(m)[1]; usados.add(j)
        if r["eggs"][j]["cls"] == e["clase"]: tot[k][0] += 1
        else: fallos.append((e["id"], "sano" if r["eggs"][j]["cls"] == 1 else "daño", round(r["eggs"][j]["conf"], 2)))
    sobran += len(r["eggs"]) - len(usados)
for k, (a, b, c) in tot.items(): print(f"{DET} conf {CONF} {k}: {a}/{b} huevos bien ({c} sin detección)")
print("detecciones de más:", sobran, "| fallos:", fallos)
