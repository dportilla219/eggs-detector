"""v5: localiza cada huevo de las imágenes del profesor (fondo blanco) y guarda cajas + hojas numeradas para revisarlas."""
import json, os, re, sys
import cv2, numpy as np
from PIL import Image

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(R, "repo_profe", "imagenes")
OUT = os.path.join(R, "v5", "profe")
os.makedirs(OUT, exist_ok=True)
nat = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def leer(p):
    im = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    return np.asarray(bg.convert("RGB"))


def cajas(img, thr):
    g = img.min(axis=2)
    m = (g < thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    H, W = g.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.004 * H * W or w < 30 or h < 30: continue
        out.append([int(x), int(y), int(x + w), int(y + h)])
    return sorted(out, key=lambda b: (round((b[1] + b[3]) / 2 / (H / 4)), b[0]))


res = {}
for sub in ("", "plantilla"):
    d = os.path.join(SRC, sub)
    for f in sorted([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))], key=nat):
        img = leer(os.path.join(d, f))
        thr = int(sys.argv[1]) if len(sys.argv) > 1 else 238
        bs = cajas(img, thr)
        res[(sub + "/" if sub else "") + f] = {"W": img.shape[1], "H": img.shape[0], "cajas": bs}
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        for k, (x1, y1, x2, y2) in enumerate(bs):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, str(k), (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(OUT, "vis_" + re.sub(r"[^A-Za-z0-9]+", "_", (sub + "_" + f).encode("ascii", "ignore").decode()) + ".jpg"), vis)
        print(f.encode("ascii", "replace").decode(), len(bs))
json.dump(res, open(os.path.join(OUT, "cajas_auto.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
