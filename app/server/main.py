"""API y servidor web del detector de huevos con daños.

Variables de entorno (todas opcionales):
    EGGS_MODEL_DIR    carpeta con los .tflite            (por defecto: ../../modelo)
    EGGS_DET_FILE     detector a usar                    (por defecto: eggs_v2_fp32.tflite)
    EGGS_SEG_FILE     modelo de zona dañada              (por defecto: eggs_dano_v1_fp16.tflite)
    EGGS_RESULTS_DIR  carpeta resultados/ del repo       (por defecto: ../../resultados)
    EGGS_SAMPLES_DIR  split YOLO para la banda (images/ + labels/), p. ej. el test
    EGGS_EVAL_JSON    salida de evaluate.py para la pestaña de métricas
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from pipeline import CONF, EggPipeline, to_rgb
from routing import ROUTES, decide_route
from samples import load_samples

try:  # fotos HEIC/HEIF de iPhone
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

log = logging.getLogger("eggs")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_DIR = os.environ.get("EGGS_MODEL_DIR", os.path.join(ROOT, "modelo"))
DET_FILE = os.environ.get("EGGS_DET_FILE", "eggs_v2_fp32.tflite")
SEG_FILE = os.environ.get("EGGS_SEG_FILE", "eggs_dano_v1_fp16.tflite")
RESULTS_DIR = os.environ.get("EGGS_RESULTS_DIR", os.path.join(ROOT, "resultados"))
SAMPLES_DIR = os.environ.get("EGGS_SAMPLES_DIR")
EVAL_JSON = os.environ.get("EGGS_EVAL_JSON")
STATIC_DIR = os.path.join(HERE, "..", "web")

MAX_UPLOAD = 15 * 1024 * 1024
MAX_SIDE = 2048                   # las fotos más grandes se reducen antes de inferir
MIN_SIDE = 32                     # por debajo no hay información suficiente para ver un huevo
Image.MAX_IMAGE_PIXELS = 80_000_000  # protege la RAM (911 MB en la instancia) de imágenes gigantes
# Como mucho 3 imágenes en proceso a la vez; el resto espera turno. Evita picos de memoria
# cuando varias personas usan la app al mismo tiempo.
SLOTS = threading.BoundedSemaphore(3)

pipe = EggPipeline(MODEL_DIR, det_file=DET_FILE, seg_file=SEG_FILE)
SAMPLES = load_samples(SAMPLES_DIR)

app = FastAPI(
    title="Detector de huevos con daños",
    description="YOLOv8n detecta y clasifica cada huevo; una U-Net (dano_v1) marca la zona dañada "
                "de los rajados y calcula su gravedad. La ruta de la banda se decide con ambos.",
    version="1.1.0",
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno al procesar la imagen. Intenta de nuevo."})


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _open(data: bytes | str) -> Image.Image:
    """Abre una imagen sin decodificarla entera si es un JPEG enorme."""
    try:
        img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
        if img.format == "JPEG":
            img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # decodifica ya reducida (1/2, 1/4, 1/8)
        img.load()
    except Image.DecompressionBombError:
        raise HTTPException(413, "La imagen tiene demasiados píxeles.")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(400, "El archivo no es una imagen válida (usa JPG, PNG, WEBP o HEIC).")
    if min(img.size) < MIN_SIDE:
        raise HTTPException(400, f"La imagen es demasiado pequeña ({img.width}×{img.height} px; mínimo {MIN_SIDE} px por lado).")
    return img


def _data_url(img: Image.Image, side: int = 1600) -> str:
    img = img.copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _run(img: Image.Image, conf: float, mode: str, return_image: bool = False) -> dict:
    if mode not in ("letterbox", "center"):
        raise HTTPException(400, "mode debe ser 'letterbox' o 'center'")
    if not 0.05 <= conf <= 0.95:
        raise HTTPException(400, "conf debe estar entre 0.05 y 0.95")
    if not SLOTS.acquire(timeout=30):
        raise HTTPException(503, "El servidor está ocupado. Intenta de nuevo en unos segundos.")
    try:
        img = to_rgb(img)
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))
        res = pipe.predict(img, conf=conf, mode=mode)
        res["route"] = decide_route(res["summary"])
        if return_image:  # para navegadores que no pueden mostrar el archivo original (p. ej. HEIC)
            res["image"] = _data_url(img)
        return res
    finally:
        SLOTS.release()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models": [pipe.det_file, pipe.seg_file], "samples": len(SAMPLES)}


@app.get("/api/info")
def info() -> dict:
    v2 = _read_json(os.path.join(RESULTS_DIR, "v2", "resumen.json"))
    v3 = _read_json(os.path.join(RESULTS_DIR, "v3", "resumen.json"))
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
        "results_v3": v3,
        "results_dano_v1": dano,
        "server_eval": ev,
        "samples": len(SAMPLES),
    }


@app.get("/api/samples")
def samples() -> list[dict]:
    return SAMPLES


def _sample(sid: int) -> dict:
    if not 0 <= sid < len(SAMPLES):
        raise HTTPException(404, "Muestra no encontrada")
    return SAMPLES[sid]


@app.get("/api/samples/{sid}/image")
def sample_image(sid: int) -> FileResponse:
    s = _sample(sid)
    return FileResponse(os.path.join(SAMPLES_DIR, "images", s["file"]),
                        headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/samples/{sid}/predict")
def predict_sample(sid: int, conf: float = CONF, mode: str = "letterbox") -> dict:
    s = _sample(sid)
    res = _run(_open(os.path.join(SAMPLES_DIR, "images", s["file"])), conf, mode)
    res["sample"] = s
    return res


@app.post("/api/predict")
def predict(file: UploadFile = File(...), conf: float = Form(CONF), mode: str = Form("letterbox"),
            return_image: bool = Form(False)) -> dict:
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "La imagen supera 15 MB.")
    if not data:
        raise HTTPException(400, "El archivo está vacío.")
    return _run(_open(data), conf, mode, return_image)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=STATIC_DIR), name="web")
