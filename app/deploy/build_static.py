"""Arma la versión sin servidor de la app: una carpeta estática que corre los mismos .tflite en el
navegador (TFLite WebAssembly, web/local.js). Sirve de respaldo cuando la instancia EC2 está apagada.

    python app/deploy/build_static.py --split ~/eggs-data/test --out dist_sin_servidor

Contenido de la carpeta:
    index.html, styles.css, app.js, local.js   interfaz (la misma de la versión con servidor)
    modelo/                                    detector + dano_v1
    wasm/                                      runtime de TFLite para el navegador (tfjs-tflite)
    datos/info.json, datos/samples.json        métricas y catálogo de la banda
    datos/muestras/                            una selección equilibrada de imágenes de test

Se puede servir con cualquier servidor de archivos estáticos (la cámara en vivo necesita HTTPS).
Con --artifact, los .tflite van en base64 dentro de archivos .txt, para visores que solo sirven tipos web.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
TFJS = "https://cdn.jsdelivr.net/npm/@tensorflow"
TFJS_CORE = f"{TFJS}/tfjs-core@4.22.0/dist/tf-core.min.js"
TFJS_CPU = f"{TFJS}/tfjs-backend-cpu@4.22.0/dist/tf-backend-cpu.min.js"
TFLITE_VER = "0.0.1-alpha.10"
TFLITE = f"{TFJS}/tfjs-tflite@{TFLITE_VER}"
WASM_SIN_HILOS = [f"tflite_web_api_cc{v}.{e}" for v in ("", "_simd") for e in ("js", "wasm")] + ["tflite_web_api_client.js"]
WASM_HILOS = [f"tflite_web_api_cc{v}.{e}" for v in ("_threaded", "_simd_threaded") for e in ("js", "wasm")] + [
    "tflite_web_api_cc_threaded.worker.js", "tflite_web_api_cc_simd_threaded.worker.js"]
TROZO_B64 = 12_000_000  # caracteres por trozo (los visores limitan cada archivo de texto a 16 MB)


def det_del_servicio() -> str:
    with open(os.path.join(HERE, "eggs-detector.service"), encoding="utf-8") as fh:
        m = re.search(r"^Environment=EGGS_DET_FILE=(\S+)", fh.read(), re.M)
    return m.group(1) if m else "eggs_v2_fp32.tflite"


def elegir_muestras(samples: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n imágenes por clase, repartidas entre el montaje y otras fuentes, en el mismo orden del test."""
    rng = random.Random(seed)
    elegidas = []
    for gt in (0, 1):
        por_fuente = {}
        for s in samples:
            if s["gt"] == gt:
                por_fuente.setdefault(s["source"], []).append(s)
        cupo = {src: max(1, round(n * len(lst) / sum(map(len, por_fuente.values())))) for src, lst in por_fuente.items()}
        for src, lst in por_fuente.items():
            elegidas += rng.sample(lst, min(cupo[src], len(lst)))
    elegidas.sort(key=lambda s: s["id"])
    return [{**s, "id": i} for i, s in enumerate(elegidas)]


def bajar(url: str, destino: str) -> None:
    if not os.path.exists(destino):
        with urllib.request.urlopen(url, timeout=120) as r, open(destino, "wb") as fh:
            fh.write(r.read())


def html_estatico(det, seg) -> str:
    with open(os.path.join(APP, "web", "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    cabeza = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
    cuerpo = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
    cabeza = cabeza.replace('initial-scale=1"', 'initial-scale=1, viewport-fit=cover"')
    cfg = {"modelos": "modelo/", "det": det, "seg": seg, "wasm": "wasm/", "datos": "datos/"}
    scripts = "\n".join([
        f'  <script src="{TFJS_CORE}"></script>',
        f'  <script src="{TFJS_CPU}"></script>',
        f'  <script src="{TFLITE}/dist/tf-tflite.min.js"></script>',
        f"  <script>window.EGGS_LOCAL = {json.dumps(cfg)};</script>",
        '  <script src="local.js"></script>',
    ])
    cuerpo = re.sub(r'(\s*<script src="app\.js[^"]*"></script>)', "\n" + scripts + r"\1", cuerpo)
    return re.sub(r"\?v=\d+", "", cabeza.strip() + "\n" + cuerpo.strip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="split YOLO de test (images/ + labels/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--det", default=det_del_servicio())
    ap.add_argument("--seg", default="eggs_dano_v1_fp16.tflite")
    ap.add_argument("--por-clase", type=int, default=40, help="imágenes de test por clase para la banda")
    ap.add_argument("--eval", help="JSON de evaluate.py ya calculado (si no, se calcula)")
    ap.add_argument("--artifact", action="store_true",
                    help="para visores que no sirven .tflite: modelos en base64 (.txt) y runtime sin hilos")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    for sub in ("modelo", "wasm", "datos/muestras"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # métricas: las mismas que devuelve /api/info en el servidor, más la evaluación del test completo
    eval_json = a.eval or os.path.join(out, "datos", "eval_test.json")
    if not a.eval and not os.path.exists(eval_json):
        subprocess.run([sys.executable, os.path.join(APP, "server", "evaluate.py"), "--split", a.split,
                        "--out", eval_json, "--det", a.det], check=True)
    os.environ.update(EGGS_DET_FILE=a.det, EGGS_SEG_FILE=a.seg, EGGS_EVAL_JSON=eval_json, EGGS_SAMPLES_DIR=a.split)
    sys.path.insert(0, os.path.join(APP, "server"))
    import main as servidor  # noqa: E402  (carga los modelos en CPU, igual que el servidor)

    info = servidor.info()
    muestras = elegir_muestras(servidor.SAMPLES, a.por_clase)
    info["samples"] = len(muestras)
    for s in muestras:
        shutil.copy2(os.path.join(a.split, "images", s["file"]), os.path.join(out, "datos", "muestras", s["file"]))
    with open(os.path.join(out, "datos", "info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False)
    with open(os.path.join(out, "datos", "samples.json"), "w", encoding="utf-8") as fh:
        json.dump(muestras, fh, ensure_ascii=False)

    cfg_modelos = []
    for f in (a.det, a.seg):
        if not a.artifact:
            shutil.copy2(os.path.join(servidor.MODEL_DIR, f), os.path.join(out, "modelo", f))
            cfg_modelos.append(f)
            continue
        with open(os.path.join(servidor.MODEL_DIR, f), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        partes = []
        for i in range(0, len(b64), TROZO_B64):
            partes.append(f"{f}.b64.{i // TROZO_B64 + 1}.txt")
            with open(os.path.join(out, "modelo", partes[-1]), "w", encoding="ascii") as fh:
                fh.write(b64[i:i + TROZO_B64])
        cfg_modelos.append({"archivo": f, "partes": partes})
    for f in WASM_SIN_HILOS + ([] if a.artifact else WASM_HILOS):
        bajar(f"{TFLITE}/wasm/{f}", os.path.join(out, "wasm", f))
    for f in ("styles.css", "app.js", "local.js"):
        shutil.copy2(os.path.join(APP, "web", f), os.path.join(out, f))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html_estatico(*cfg_modelos))

    tam = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f"Listo: {out} ({tam / 1e6:.1f} MB, {len(muestras)} imágenes de test, detector {a.det})")


if __name__ == "__main__":
    main()
