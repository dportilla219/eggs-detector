"""API y servidor web del detector de huevos con daños.

Variables de entorno (todas opcionales):
    EGGS_MODEL_DIR    carpeta con los .tflite            (por defecto: ../../modelo)
    EGGS_RESULTS_DIR  carpeta resultados/ del repo       (por defecto: ../../resultados)
    EGGS_SAMPLES_DIR  split YOLO para la banda (images/ + labels/), p. ej. el test
    EGGS_EVAL_JSON    salida de evaluate.py para la pestaña de métricas
"""

from __future__ import annotations

import io
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from pipeline import CONF, EggPipeline
from routing import ROUTES, decide_route
from samples import load_samples

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_DIR = os.environ.get("EGGS_MODEL_DIR", os.path.join(ROOT, "modelo"))
RESULTS_DIR = os.environ.get("EGGS_RESULTS_DIR", os.path.join(ROOT, "resultados"))
SAMPLES_DIR = os.environ.get("EGGS_SAMPLES_DIR")
EVAL_JSON = os.environ.get("EGGS_EVAL_JSON")
STATIC_DIR = os.path.join(HERE, "..", "web")

MAX_UPLOAD = 12 * 1024 * 1024
MAX_SIDE = 2048  # las fotos más grandes se reducen antes de inferir

pipe = EggPipeline(MODEL_DIR)
SAMPLES = load_samples(SAMPLES_DIR)

app = FastAPI(
    title="Detector de huevos con daños",
    description="YOLOv8n (v2) detecta y clasifica cada huevo; una U-Net (dano_v1) marca la zona dañada "
                "de los rajados y calcula su gravedad. La ruta de la banda se decide con ambos.",
    version="1.0.0",
)


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _run(img: Image.Image, conf: float, mode: str) -> dict:
    if mode not in ("letterbox", "center"):
        raise HTTPException(400, "mode debe ser 'letterbox' o 'center'")
    if not 0.05 <= conf <= 0.95:
        raise HTTPException(400, "conf debe estar entre 0.05 y 0.95")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    res = pipe.predict(img, conf=conf, mode=mode)
    res["route"] = decide_route(res["summary"])
    return res


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models": [pipe.det_file, pipe.seg_file], "samples": len(SAMPLES)}


@app.get("/api/info")
def info() -> dict:
    v2 = _read_json(os.path.join(RESULTS_DIR, "v2", "resumen.json"))
    dano = _read_json(os.path.join(RESULTS_DIR, "dano_v1", "resumen.json"))
    if dano:  # el detalle por huevo es largo y no hace falta en la interfaz
        dano = {k: v for k, v in dano.items() if k != "pipeline_detalle"}
    ev = _read_json(EVAL_JSON) if EVAL_JSON else None
    if ev:
        ev = {k: v for k, v in ev.items() if k != "detalle"}
    sizes = {f: os.path.getsize(os.path.join(MODEL_DIR, f)) for f in (pipe.det_file, pipe.seg_file)}
    return {
        "models": {
            "detector": {"file": pipe.det_file, "bytes": sizes[pipe.det_file], "input": [1, 640, 640, 3],
                         "output": [1, 6, 8400], "classes": ["Crack", "Intact"], "conf": CONF, "iou_nms": 0.5},
            "damage": {"file": pipe.seg_file, "bytes": sizes[pipe.seg_file], "input": [1, 192, 192, 3],
                       "output": [1, 192, 192, 2], "levels": {"leve": "< 15 %", "media": "15–35 %", "grave": "≥ 35 %"}},
        },
        "routes": ROUTES,
        "results_v2": v2,
        "results_dano_v1": dano,
        "server_eval": ev,
        "samples": len(SAMPLES),
    }


@app.get("/api/samples")
def samples() -> list[dict]:
    return SAMPLES


@app.get("/api/samples/{sid}/image")
def sample_image(sid: int) -> FileResponse:
    if not 0 <= sid < len(SAMPLES):
        raise HTTPException(404, "muestra no encontrada")
    return FileResponse(os.path.join(SAMPLES_DIR, "images", SAMPLES[sid]["file"]),
                        headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/samples/{sid}/predict")
def predict_sample(sid: int, conf: float = CONF, mode: str = "letterbox") -> dict:
    if not 0 <= sid < len(SAMPLES):
        raise HTTPException(404, "muestra no encontrada")
    s = SAMPLES[sid]
    res = _run(Image.open(os.path.join(SAMPLES_DIR, "images", s["file"])), conf, mode)
    res["sample"] = s
    return res


@app.post("/api/predict")
def predict(file: UploadFile = File(...), conf: float = Form(CONF), mode: str = Form("letterbox")) -> dict:
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "La imagen supera 12 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "El archivo no es una imagen válida")
    return _run(img, conf, mode)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=STATIC_DIR), name="web")
