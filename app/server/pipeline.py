"""Tubería de inferencia: v2 (detector) -> recorte -> dano_v1 (zona dañada).

Sigue la especificación de MODELO_IO.md del repositorio:
- v2:      entrada [1,640,640,3] float32 RGB 0..1, salida [1,6,8400] (cx,cy,w,h,Crack,Intact), sin NMS.
- dano_v1: entrada [1,192,192,3] float32 RGB 0..1 (caja +5 % por lado, estirada),
           salida [1,192,192,2] (canal 0 = silueta del huevo, canal 1 = zona dañada).
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

try:  # runtime oficial actual de TFLite
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover - alternativas
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore

CLASSES = ("Crack", "Intact")
DET_SIZE = 640
SEG_SIZE = 192
CONF = 0.5
IOU = 0.5
CROP_PAD = 0.05  # 5 % por lado = caja +10 %
LEVELS = ((0.15, "leve"), (0.35, "media"), (1.01, "grave"))

# Color del daño en la máscara (magenta, como en resultados/dano_v1/ejemplos_test.jpg)
DMG_RGB = (236, 64, 200)


@dataclass
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float


def _load(path: str, threads: int) -> Interpreter:
    interp = Interpreter(model_path=path, num_threads=threads)
    interp.allocate_tensors()
    return interp


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU de una caja `a` [4] contra varias `b` [N,4] (x1,y1,x2,y2)."""
    iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def to_rgb(img: Image.Image) -> Image.Image:
    """Cualquier imagen que abra Pillow -> RGB 8 bits, derecha según su EXIF."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):  # PNG/TIFF de 16 bits o flotante
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        img = Image.fromarray(np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), "L")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")  # la transparencia se rellena de blanco, no de negro
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def severity_level(sev: float) -> str:
    for limit, name in LEVELS:
        if sev < limit:
            return name
    return "grave"


class EggPipeline:
    def __init__(self, model_dir: str, det_file: str = "eggs_v2_fp32.tflite",
                 seg_file: str = "eggs_dano_v1_fp16.tflite", threads: int | None = None):
        threads = threads or max(1, min(4, os.cpu_count() or 1))
        self.det_file, self.seg_file = det_file, seg_file
        self.det = _load(os.path.join(model_dir, det_file), threads)
        self.seg = _load(os.path.join(model_dir, seg_file), threads)
        self._lock = threading.Lock()  # los intérpretes TFLite no son thread-safe
        self._check_io()
        self._warmup()

    def _warmup(self) -> None:
        """Una inferencia en vacío al arrancar: así la primera petición real no paga la inicialización."""
        self.det.set_tensor(self._det_in, np.zeros((1, DET_SIZE, DET_SIZE, 3), np.float32))
        self.det.invoke()
        self.seg.set_tensor(self._seg_in, np.zeros((1, SEG_SIZE, SEG_SIZE, 3), np.float32))
        self.seg.invoke()

    def _check_io(self) -> None:
        di, do = self.det.get_input_details()[0], self.det.get_output_details()[0]
        si, so = self.seg.get_input_details()[0], self.seg.get_output_details()[0]
        assert tuple(di["shape"]) == (1, DET_SIZE, DET_SIZE, 3), di["shape"]
        assert tuple(do["shape"]) == (1, 6, 8400), do["shape"]
        assert tuple(si["shape"]) == (1, SEG_SIZE, SEG_SIZE, 3), si["shape"]
        assert tuple(so["shape"]) == (1, SEG_SIZE, SEG_SIZE, 2), so["shape"]
        self._det_in, self._det_out = di["index"], do["index"]
        self._seg_in, self._seg_out = si["index"], so["index"]

    # ------------------------------------------------------------------ v2
    def _det_input(self, img: Image.Image, mode: str) -> tuple[np.ndarray, Letterbox, tuple[int, int, int]]:
        """Prepara la entrada de v2.

        mode="letterbox": imagen completa escalada y rellenada con gris (114), como en el
        entrenamiento de Ultralytics. No pierde nada de la foto.
        mode="center": cuadrado centrado, como el frame processor de la app móvil.
        Devuelve también el cuadrado de origen (x, y, lado) en píxeles de la imagen.
        """
        w, h = img.size
        if mode == "center":
            side = min(w, h)
            ox, oy = (w - side) // 2, (h - side) // 2
            sq = img.crop((ox, oy, ox + side, oy + side)).resize((DET_SIZE, DET_SIZE), Image.BILINEAR)
            arr = np.asarray(sq, dtype=np.float32) / 255.0
            return arr[None], Letterbox(DET_SIZE / side, -ox * DET_SIZE / side, -oy * DET_SIZE / side), (ox, oy, side)
        scale = DET_SIZE / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = Image.new("RGB", (DET_SIZE, DET_SIZE), (114, 114, 114))
        px, py = (DET_SIZE - nw) // 2, (DET_SIZE - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return arr[None], Letterbox(scale, px, py), (0, 0, max(w, h))

    def _decode(self, out: np.ndarray, conf_thr: float) -> list[tuple[np.ndarray, float, int, float, float]]:
        """Decodifica [6,8400] -> lista (caja 0..640, conf, clase, sCrack, sIntact) tras NMS agnóstica."""
        s_crack, s_intact = out[4], out[5]
        conf = np.maximum(s_crack, s_intact)
        keep = conf >= conf_thr
        if not keep.any():
            return []
        cx, cy, bw, bh = (out[i][keep] * DET_SIZE for i in range(4))
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        conf, sc, si = conf[keep], s_crack[keep], s_intact[keep]
        order = np.argsort(-conf)
        boxes, conf, sc, si = boxes[order], conf[order], sc[order], si[order]
        chosen: list[int] = []
        for i in range(len(boxes)):
            if not chosen or (_iou(boxes[i], boxes[chosen]) < IOU).all():
                chosen.append(i)
        return [(boxes[i], float(conf[i]), 0 if sc[i] >= si[i] else 1, float(sc[i]), float(si[i])) for i in chosen]

    # ------------------------------------------------------------- dano_v1
    def _segment(self, img: Image.Image, box: tuple[float, float, float, float]) -> dict:
        """Recorta el huevo del frame completo (+5 % por lado), lo estira a 192x192 y segmenta."""
        W, H = img.size
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        rx, ry = max(0.0, x1 - CROP_PAD * bw), max(0.0, y1 - CROP_PAD * bh)
        rx2, ry2 = min(float(W), x2 + CROP_PAD * bw), min(float(H), y2 + CROP_PAD * bh)
        rw, rh = rx2 - rx, ry2 - ry
        crop = img.crop((rx, ry, rx2, ry2)).resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)  # estirado
        arr = np.asarray(crop, dtype=np.float32)[None] / 255.0
        self.seg.set_tensor(self._seg_in, arr)
        self.seg.invoke()
        out = self.seg.get_tensor(self._seg_out)[0]  # [192,192,2]
        egg_p, dmg_p = out[..., 0], out[..., 1]
        egg = egg_p > 0.5
        dmg = (dmg_p > 0.5) & egg  # solo daño dentro de la silueta
        egg_px, dmg_px = int(egg.sum()), int(dmg.sum())
        sev = dmg_px / egg_px if egg_px else 0.0
        return {
            "rect": [round(rx, 1), round(ry, 1), round(rw, 1), round(rh, 1)],
            "severity": round(sev, 4),
            "level": severity_level(sev),
            "egg_px": egg_px,
            "damage_px": dmg_px,
            "mask_png": self._mask_png(dmg_p * egg, rw, rh),
            "crop_png": _to_data_url(crop, "JPEG"),
        }

    @staticmethod
    def _mask_png(dmg_p: np.ndarray, rw: float, rh: float) -> str:
        """Máscara RGBA del daño, reescalada al tamaño del recorte (máx. 384 px) para bordes suaves."""
        scale = min(1.0, 384 / max(rw, rh, 1))
        tw, th = max(8, round(rw * scale)), max(8, round(rh * scale))
        prob = Image.fromarray((np.clip(dmg_p, 0, 1) * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
        p = np.asarray(prob, dtype=np.float32) / 255.0
        m = p > 0.5
        inner = m.copy()
        inner[1:, :] &= m[:-1, :]
        inner[:-1, :] &= m[1:, :]
        inner[:, 1:] &= m[:, :-1]
        inner[:, :-1] &= m[:, 1:]
        edge = m & ~inner
        rgba = np.zeros((th, tw, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = DMG_RGB
        rgba[..., 3] = np.where(m, 120, 0)
        rgba[..., 3][edge] = 255
        return _to_data_url(Image.fromarray(rgba, "RGBA"), "PNG")

    # ---------------------------------------------------------------- API
    def predict(self, img: Image.Image, conf: float = CONF, mode: str = "letterbox",
                with_damage: bool = True) -> dict:
        img = to_rgb(img)
        W, H = img.size
        t0 = time.perf_counter()
        x, lb, square = self._det_input(img, mode)
        with self._lock:
            self.det.set_tensor(self._det_in, x)
            self.det.invoke()
            raw = self.det.get_tensor(self._det_out)[0]
            t_det = time.perf_counter()
            dets = self._decode(raw, conf)
            eggs = []
            for box, c, cls, sc, si in dets:
                # de coordenadas de 640 a píxeles de la imagen original
                bx = (box - np.array([lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y])) / lb.scale
                bx = np.clip(bx, [0, 0, 0, 0], [W, H, W, H])
                if bx[2] - bx[0] < 2 or bx[3] - bx[1] < 2:
                    continue
                egg = {
                    "box": [round(float(v), 1) for v in bx],
                    "cls": cls,
                    "label": CLASSES[cls],
                    "conf": round(c, 4),
                    "scores": {"Crack": round(sc, 4), "Intact": round(si, 4)},
                    "damage": None,
                }
                if cls == 0 and with_damage:  # dano_v1 solo sobre huevos rajados
                    egg["damage"] = self._segment(img, tuple(float(v) for v in bx))
                eggs.append(egg)
            t_end = time.perf_counter()
        n_crack = sum(e["cls"] == 0 for e in eggs)
        worst = max((e["damage"]["severity"] for e in eggs if e["damage"]), default=None)
        return {
            "width": W,
            "height": H,
            "mode": mode,
            "square": list(square),
            "eggs": eggs,
            "summary": {
                "n_eggs": len(eggs),
                "n_crack": n_crack,
                "n_intact": len(eggs) - n_crack,
                "max_severity": worst,
                "max_level": severity_level(worst) if worst is not None else None,
            },
            "timing_ms": {
                "detector": round((t_det - t0) * 1000, 1),
                "damage": round((t_end - t_det) * 1000, 1),
                "total": round((t_end - t0) * 1000, 1),
            },
        }


def _to_data_url(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 85} if fmt == "JPEG" else {"optimize": True}))
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()
