"""Pelotas de ping-pong enteras (Commons): cuántas ve cada detector y si las da por sanas."""
import os, sys
from PIL import Image
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(R, "repo", "app", "server"))
from pipeline import EggPipeline, to_rgb  # noqa: E402
IDS = [2, 4, 7, 17, 20, 21, 23, 24, 26, 27, 28, 34, 35, 39, 41, 43, 49, 93, 101]  # no usadas para entrenar v5b
p = EggPipeline(os.path.join(R, "repo", "modelo"), det_file=sys.argv[1])
s = d = 0; im_ok = 0; det = []
for i in IDS:
    r = p.predict(to_rgb(Image.open(os.path.join(R, "v5", "pelotas", "cand", f"{i:04d}.jpg"))), conf=0.5, with_damage=False)
    c = [e["cls"] for e in r["eggs"]]; s += c.count(1); d += c.count(0); im_ok += bool(c) and 0 not in c; det.append((i, c.count(1), c.count(0)))
print(sys.argv[1], f"pelotas detectadas: {s} sanas, {d} con daño | fotos con todo sano: {im_ok}/{len(IDS)}")
print([x for x in det if x[2] or not (x[1] + x[2])])
