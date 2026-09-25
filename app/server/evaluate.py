"""Evalúa la tubería completa (v2 + dano_v1) sobre un split YOLO y guarda un resumen JSON.

Uso:
    python evaluate.py --split ~/eggs-data/test --out ~/eggs-data/eval_test.json

La app muestra ese JSON en la pestaña "Modelo y métricas".
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import EggPipeline  # noqa: E402
from routing import decide_route  # noqa: E402
from samples import load_samples  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=os.path.join(HERE, "..", "..", "modelo"))
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--det", default=os.environ.get("EGGS_DET_FILE", "eggs_v2_fp32.tflite"))
    a = ap.parse_args()

    pipe = EggPipeline(a.models, det_file=a.det)
    items = load_samples(a.split)
    rows = []
    t0 = time.time()
    for it in items:
        r = pipe.predict(Image.open(os.path.join(a.split, "images", it["file"])), conf=a.conf)
        s = r["summary"]
        pred = None if s["n_eggs"] == 0 else (0 if s["n_crack"] else 1)
        rows.append({
            "file": it["file"], "source": it["source"], "gt": it["gt"], "pred": pred,
            "severity": s["max_severity"], "route": decide_route(s), "ms": r["timing_ms"]["total"],
        })

    def group(name: str, sub: list[dict], gt: int) -> dict:
        return {
            "grupo": name, "n": len(sub),
            "acierto": round(sum(x["pred"] == gt for x in sub) / len(sub), 4) if sub else None,
            "sin_deteccion": sum(x["pred"] is None for x in sub),
        }

    groups = []
    for src, label in (("montaje", "del montaje"), ("otras", "de otras fuentes")):
        for gt, cls in ((0, "Crack"), (1, "Intact")):
            sub = [x for x in rows if x["source"] == src and x["gt"] == gt]
            if sub:
                groups.append(group(f"{cls} {label}", sub, gt))
    for gt, cls in ((0, "Crack"), (1, "Intact")):
        groups.append(group(f"{cls} (total)", [x for x in rows if x["gt"] == gt], gt))

    confusion = {real: {p: 0 for p in ("Crack", "Intact", "Ninguno")} for real in ("Crack", "Intact")}
    for x in rows:
        real = "Crack" if x["gt"] == 0 else "Intact"
        pred = {0: "Crack", 1: "Intact", None: "Ninguno"}[x["pred"]]
        confusion[real][pred] += 1

    ms = sorted(x["ms"] for x in rows)
    routes: dict[str, int] = {}
    for x in rows:
        routes[x["route"]] = routes.get(x["route"], 0) + 1
    sev_crack = [x["severity"] for x in rows if x["pred"] == 0 and x["severity"] is not None]
    out = {
        "split": os.path.basename(os.path.normpath(a.split)),
        "n_imagenes": len(rows),
        "conf": a.conf,
        "modelos": [pipe.det_file, pipe.seg_file],
        "acierto_global": round(sum(x["pred"] == x["gt"] for x in rows) / len(rows), 4),
        "grupos": groups,
        "confusion": confusion,
        "rutas": routes,
        "gravedad_media_crack": round(statistics.mean(sev_crack), 4) if sev_crack else None,
        "latencia_ms": {"p50": ms[len(ms) // 2], "p90": ms[int(len(ms) * 0.9)], "media": round(statistics.mean(ms), 1)},
        "duracion_s": round(time.time() - t0, 1),
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "detalle": rows,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "detalle"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
