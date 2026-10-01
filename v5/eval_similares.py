"""Prueba independiente pequeña (Commons, etiquetada a mano): cuántos huevos ve cada modelo y de qué clase."""
import json, os, sys
from PIL import Image
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(R, "repo", "app", "server"))
from pipeline import EggPipeline, to_rgb  # noqa: E402
SANOS = [417, 418, 419, 420, 421, 422, 423, 424, 426, 427, 428, 429, 431, 455, 458, 464, 520, 521, 522, 537, 5, 9]
DANO = [11, 294, 296]
p = EggPipeline(os.path.join(R, "repo", "modelo"), det_file=sys.argv[1]); conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
ok = {1: 0, 0: 0}; det = []
for gt, ids in ((1, SANOS), (0, DANO)):
    for i in ids:
        img = to_rgb(Image.open(os.path.join(R, "v5", "similares", "cand", f"{i:04d}.jpg")))
        r = p.predict(img, conf=conf, with_damage=False)
        c = [e["cls"] for e in r["eggs"]]
        bien = (gt == 1 and c and 0 not in c) or (gt == 0 and 0 in c)
        ok[gt] += bool(bien); det.append((i, "sano" if gt else "daño", f"{c.count(0)} daño/{c.count(1)} sano", "OK" if bien else "MAL"))
print(sys.argv[1], f"sanos {ok[1]}/{len(SANOS)} | con daño {ok[0]}/{len(DANO)}")
print([d for d in det if d[3] == "MAL"])
