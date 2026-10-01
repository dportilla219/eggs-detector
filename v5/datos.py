"""Datos de v5: composiciones con los huevos del repositorio del profesor (adiacla/huevos), huevos sanos equivalentes
y manchas sintéticas. Sirve para el detector (formato YOLO) y para el modelo de zona dañada (recorte + máscaras).

Clases del detector: 0 = Crack (huevo con daño: rajado, roto, deformado, sucio, con moho o manchas), 1 = Intact (sano).

Por qué hace falta fabricar sanos: en el repositorio del profesor casi todos los huevos tienen daño. Si el modelo solo viera
esos, aprendería "huevo de esa colección = daño". Por eso cada huevo con daño localizado se "cura" (se rellena la zona
con textura de un huevo sano) y entra también como sano, junto a huevos sanos reales recortados.
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

SEG = 192
# Huevos del profesor que, una vez "curados", quedan como un huevo sano creíble (revisados a ojo en una hoja).
CURADOS_OK = {0, 4, 7, 8, 9, 12, 17, 18, 22, 24, 27, 30, 35, 38, 39, 40, 43, 49, 51, 56, 61, 62, 65, 78, 80, 88, 92, 95,
              120, 124, 128, 136, 140, 145, 148, 100, 104, 105, 107, 110, 112, 114, 118}


# ------------------------------------------------------------------ utilidades
def leer_rgb(p):
    a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(p)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 4:  # transparencia sobre blanco
        al = a[..., 3:4].astype(np.float32) / 255
        a = (a[..., :3].astype(np.float32) * al + 255 * (1 - al)).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def ruido(h, w, escala, rng):
    """Campo aleatorio suave en 0..1 (ruido gaussiano filtrado)."""
    k = max(2, int(max(h, w) * escala))
    f = rng.standard_normal((max(2, h // k + 2), max(2, w // k + 2))).astype(np.float32)
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
    return (f - f.min()) / (f.max() - f.min() + 1e-6)


# ------------------------------------------------------------------ banco de huevos
def cargar_profe(profe_dir, anot_dir):
    info = json.load(open(os.path.join(anot_dir, "profe_huevos.json"), encoding="utf-8"))
    huevos, originales = [], []
    for f, d in info.items():
        img = leer_rgb(os.path.join(profe_dir, f))
        cajas = []
        for e in d["huevos"]:
            x1, y1, x2, y2 = e["caja"]
            if e["omitir"]:
                img[y1:y2, x1:x2] = 255  # no es un huevo entero: se borra de la imagen original
                continue
            sil = cv2.imread(os.path.join(anot_dir, "masks", f"{e['id']:03d}_sil.png"), 0) > 127
            dano = cv2.imread(os.path.join(anot_dir, "masks", f"{e['id']:03d}_dano.png"), 0) > 127
            rgb = img[y1:y2, x1:x2].copy()
            blanco = float(rgb.min(axis=2)[sil].mean()) > 170
            huevos.append({"id": e["id"], "rgb": rgb, "sil": sil, "dano": dano, "clase": e["clase"], "origen": "profe",
                           "blanco": blanco, "sucio": e.get("sucio", False),
                           "curable": e["id"] in CURADOS_OK})
            cajas.append((e["clase"], x1, y1, x2, y2))
        originales.append({"nombre": f, "rgb": img, "cajas": cajas})
    return huevos, originales


def curar(e, donante, rng):
    """Huevo sano a partir de uno con daño localizado: clonado de Poisson de la textura de un huevo sano sobre la zona."""
    h, w = e["sil"].shape
    P = 12
    dst = cv2.copyMakeBorder(e["rgb"], P, P, P, P, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    src = cv2.resize(donante["rgb"], (w, h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        src = src[:, ::-1]
    if e["blanco"]:  # huevo blanco: la textura del donante sin su color
        g = src.astype(np.float32).mean(axis=2, keepdims=True)
        src = np.clip(np.repeat(g, 3, 2) * (float(np.median(e["rgb"][e["sil"]])) / max(1.0, float(np.median(g)))), 0, 255).astype(np.uint8)
        src = cv2.GaussianBlur(src, (0, 0), 2.0)
    src = cv2.copyMakeBorder(np.ascontiguousarray(src), P, P, P, P, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    k = max(5, int(0.07 * max(h, w))) | 1
    m = cv2.dilate(e["dano"].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m &= cv2.erode(e["sil"].astype(np.uint8), np.ones((5, 5), np.uint8))
    m = cv2.copyMakeBorder(m * 255, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    ys, xs = np.where(m > 0)
    if len(xs) < 20:
        return None
    centro = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))
    out = cv2.seamlessClone(src, dst, m, centro, cv2.NORMAL_CLONE)[P:-P, P:-P]
    return {"id": e["id"], "rgb": out, "sil": e["sil"], "dano": np.zeros_like(e["dano"]), "clase": 1, "origen": "curado",
            "blanco": e["blanco"], "sucio": False, "curable": False}


def cargar_reales(lista, tflite, max_n, rng, lado=360):
    """Huevos de fotos reales recortados con la silueta que da el modelo de zona dañada (canal 0)."""
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tflite)
    it.allocate_tensors()
    ii, oo = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    out = []
    lista = [x for x in lista if x.get("split", "train") == "train"]
    rng.shuffle(lista)
    por_clase = {0: 0, 1: 0}
    for x in lista:
        if por_clase[x["clase"]] >= max_n:
            continue
        try:
            img = leer_rgb(x["file"])
        except Exception:
            continue
        H, W = img.shape[:2]
        for (x1, y1, x2, y2) in x["cajas"][:2]:
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < 90:
                continue
            b = [int(max(0, x1 - 0.1 * bw)), int(max(0, y1 - 0.1 * bh)), int(min(W, x2 + 0.1 * bw)), int(min(H, y2 + 0.1 * bh))]
            crop = img[b[1]:b[3], b[0]:b[2]]
            it.set_tensor(ii, (cv2.resize(crop, (SEG, SEG), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255)[None])
            it.invoke()
            o = it.get_tensor(oo)[0]
            k = min(1.0, lado / max(crop.shape[:2]))
            crop = cv2.resize(crop, (max(8, int(crop.shape[1] * k)), max(8, int(crop.shape[0] * k))), interpolation=cv2.INTER_AREA)
            sil = cv2.resize(o[..., 0], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
            dano = (cv2.resize(o[..., 1], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5) & sil
            n, lab, st, _ = cv2.connectedComponentsWithStats(sil.astype(np.uint8))
            if n < 2:
                continue
            sil = lab == 1 + np.argmax(st[1:, 4])
            k = int(0.3 * min(sil.shape)) | 1  # quita colas de sombra: un huevo es convexo y ancho
            sil = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
            if sil.sum() < 100:
                continue
            ys, xs = np.where(sil)
            llena = sil.sum() / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
            toca = xs.min() == 0 or ys.min() == 0 or xs.max() == sil.shape[1] - 1 or ys.max() == sil.shape[0] - 1
            if not (0.68 < llena < 0.86) or toca or sil.sum() < 0.3 * sil.size:
                continue  # silueta dudosa o huevo cortado
            sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            out.append({"id": -1, "rgb": crop[sl].copy(), "sil": sil[sl].copy(), "dano": (dano[sl] if x["clase"] == 0 else np.zeros_like(sil[sl])),
                        "clase": x["clase"], "origen": "real", "blanco": False, "sucio": False, "curable": False})
            por_clase[x["clase"]] += 1
    return out


# ------------------------------------------------------------------ manchas sintéticas
def ensuciar(e, rng):
    """Mancha, moho, suciedad o salpicado sintético sobre un huevo sano. Devuelve el huevo con daño y su máscara exacta."""
    rgb, sil = e["rgb"].astype(np.float32), e["sil"]
    h, w = sil.shape
    tipo = rng.choice(["mancha", "moho", "barro", "pecas", "tinta", "polvo", "oxido"])
    dentro = cv2.erode(sil.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def blob(n=1, rmin=0.12, rmax=0.3, borde=0.5):
        a = np.zeros((h, w), np.float32)
        for _ in range(n):
            cx, cy = w * rng.uniform(0.25, 0.75), h * rng.uniform(0.25, 0.75)
            rx, ry = w * rng.uniform(rmin, rmax), h * rng.uniform(rmin, rmax)
            d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) + (ruido(h, w, 0.12, rng) - 0.5) * 0.9
            a = np.maximum(a, np.clip((1 - d) / borde, 0, 1))
        return a

    if tipo == "mancha":      # mancha oscura difusa (golpe, quemadura)
        a = blob(rng.integers(1, 3), 0.12, 0.32, 0.6) * rng.uniform(0.55, 0.9)
        color = np.array([rng.uniform(20, 70), rng.uniform(15, 50), rng.uniform(10, 45)])
    elif tipo == "moho":      # disco verdoso/gris con textura
        a = blob(rng.integers(1, 6), 0.06, 0.22, 0.45) * (0.55 + 0.45 * ruido(h, w, 0.02, rng)) * rng.uniform(0.7, 0.95)
        color = np.array([rng.uniform(70, 130), rng.uniform(110, 150), rng.uniform(90, 130)])
    elif tipo == "barro":     # suciedad en una franja del huevo
        lim = rng.uniform(0.45, 0.7)
        franja = np.clip((yy / h - lim + (ruido(h, w, 0.15, rng) - 0.5) * 0.3) / 0.08, 0, 1)
        if rng.random() < 0.25:
            franja = franja[::-1]
        a = franja * (0.6 + 0.4 * ruido(h, w, 0.03, rng)) * rng.uniform(0.8, 1.0)
        base = np.array([rng.uniform(60, 120), rng.uniform(45, 95), rng.uniform(25, 70)])
        color = base[None, None] * (0.6 + 0.8 * ruido(h, w, 0.015, rng)[..., None])
    elif tipo == "pecas":     # salpicado por todo el huevo
        r = ruido(h, w, 0.012, rng) * ruido(h, w, 0.03, rng)
        a = np.clip((r - np.quantile(r, rng.uniform(0.86, 0.93))) / 0.03, 0, 1) * rng.uniform(0.7, 1.0)
        color = np.array([[rng.uniform(40, 150), rng.uniform(20, 60), rng.uniform(15, 50)], [30, 30, 30], [60, 110, 70]][rng.integers(0, 3)])
    elif tipo == "tinta":     # borrón oscuro de bordes duros
        a = (blob(1, 0.1, 0.25, 0.15) * (ruido(h, w, 0.02, rng) > 0.42)).astype(np.float32) * rng.uniform(0.8, 1.0)
        color = np.array([rng.uniform(10, 40)] * 3)
    elif tipo == "polvo":     # costra o polvo claro
        a = blob(rng.integers(1, 3), 0.12, 0.3, 0.3) * (0.5 + 0.5 * ruido(h, w, 0.015, rng)) * rng.uniform(0.75, 0.95)
        color = np.array([rng.uniform(225, 250), rng.uniform(220, 245), rng.uniform(205, 235)])
        if e["blanco"]:
            color = color * np.array([0.8, 0.74, 0.6])
    else:                     # óxido: chorreado marrón rojizo
        r = ruido(h, w, 0.05, rng)
        r = cv2.GaussianBlur(r, (1, 2 * int(0.08 * h) + 1), 0)
        a = np.clip((r - np.quantile(r, 0.7)) / 0.05, 0, 1) * blob(1, 0.3, 0.45, 0.4) * rng.uniform(0.6, 0.9)
        color = np.array([rng.uniform(110, 160), rng.uniform(55, 85), rng.uniform(15, 40)])
    a = (a * dentro)[..., None]
    out = np.clip(rgb * (1 - a) + color * a, 0, 255).astype(np.uint8)
    m = (a[..., 0] > 0.25)
    if tipo == "pecas":
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((int(0.12 * w) | 1, int(0.12 * w) | 1), np.uint8)).astype(bool) & sil
    if m.sum() < 0.02 * sil.sum():
        return None
    return {"id": e["id"], "rgb": out, "sil": sil, "dano": m, "clase": 0, "origen": "manchado", "blanco": e["blanco"],
            "sucio": True, "curable": False}


def armar_banco(profe_dir, anot_dir, reales, tflite, rng, max_reales=260, ver=print):
    profe, originales = cargar_profe(profe_dir, anot_dir)
    sanos_ia = [e for e in profe if e["clase"] == 1]
    curados = []
    for e in profe:
        if e["curable"]:
            for _ in range(2):
                c = curar(e, sanos_ia[rng.integers(0, len(sanos_ia))], rng)
                if c is not None:
                    curados.append(c)
    rl = cargar_reales(reales, tflite, max_reales, rng) if reales else []
    banco = {"dano_profe": [e for e in profe if e["clase"] == 0], "sano_ia": sanos_ia + curados,
             "sano_real": [e for e in rl if e["clase"] == 1], "dano_real": [e for e in rl if e["clase"] == 0]}
    ver({k: len(v) for k, v in banco.items()})
    return banco, originales


# ------------------------------------------------------------------ composición
def variar(e, rng, lado):
    """Escala, giro, espejo y cambio leve de luz. Devuelve rgb, silueta (0..1) y máscara de daño."""
    rgb, sil, dano = e["rgb"], e["sil"].astype(np.uint8) * 255, e["dano"].astype(np.uint8) * 255
    if rng.random() < 0.5:
        rgb, sil, dano = rgb[:, ::-1], sil[:, ::-1], dano[:, ::-1]
    h, w = sil.shape
    k = lado / max(h, w)
    ang = rng.normal(0, 7) if rng.random() < 0.8 else rng.uniform(-180, 180)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, k)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 4, int(h * c + w * s) + 4
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rgb = cv2.warpAffine(np.ascontiguousarray(rgb), M, (nw, nh), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    sil = cv2.warpAffine(np.ascontiguousarray(sil), M, (nw, nh), flags=cv2.INTER_LINEAR)
    dano = cv2.warpAffine(np.ascontiguousarray(dano), M, (nw, nh), flags=cv2.INTER_LINEAR)
    f = rgb.astype(np.float32) * rng.uniform(0.86, 1.1) + rng.uniform(-10, 10)
    f = f * (1 + rng.uniform(-0.04, 0.04, 3))
    return np.clip(f, 0, 255).astype(np.uint8), sil.astype(np.float32) / 255, dano > 127


def fondo(H, W, rng):
    t = rng.random()
    if t < 0.62:
        v = rng.uniform(244, 255)
        f = np.full((H, W, 3), v, np.float32)
        return f, True
    if t < 0.74:   # gris claro con degradado
        g = np.linspace(rng.uniform(200, 250), rng.uniform(200, 250), H, dtype=np.float32)[:, None, None]
        return np.repeat(np.repeat(g, W, 1), 3, 2), False
    if t < 0.9:    # color liso (mesa, cinta, cartón)
        c = np.array([[30, 30, 30], [90, 60, 40], [40, 70, 110], [120, 120, 120], [200, 180, 140], [20, 90, 50]][rng.integers(0, 6)], np.float32)
        return np.full((H, W, 3), 1, np.float32) * (c + rng.uniform(-15, 15, 3)), False
    n = ruido(H, W, 0.01, rng)[..., None] * 60 + ruido(H, W, 0.2, rng)[..., None] * 60 + rng.uniform(40, 130)
    return np.repeat(n, 3, 2) * (1 + rng.uniform(-0.1, 0.1, 3)), False


def pegar(lienzo, rgb, sil, x, y, blanco, sombra, rng):
    h, w = sil.shape
    H, W = lienzo.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return False
    a = sil.copy()
    if not blanco:  # fuera del blanco el borde del recorte deja un halo claro: se come 1-2 px y se suaviza
        a = cv2.erode(a, np.ones((3, 3), np.uint8))
    a = cv2.GaussianBlur(a, (5, 5), 0)[..., None]
    reg = lienzo[y:y + h, x:x + w]
    if sombra:
        sh = cv2.GaussianBlur(np.roll(sil, int(0.03 * h), axis=0), (0, 0), max(2, 0.04 * w))[..., None]
        reg *= 1 - 0.25 * sh
    reg[:] = reg * (1 - a) + rgb.astype(np.float32) * a
    return True


def pantalla(img, cajas, rng):
    """Como si se le tomara foto a una pantalla o a una hoja: perspectiva leve, desenfoque, brillo y compresión."""
    H, W = img.shape[:2]
    d = 0.06
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    m = rng.uniform(0.03, 0.12)
    dst = src * (1 - 2 * m) + np.float32([W * m, H * m]) + rng.uniform(-d, d, (4, 2)).astype(np.float32) * np.float32([W, H]) * 0.5
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    borde = [int(v) for v in rng.uniform(0, 90, 3)]
    out = cv2.warpPerspective(img, M, (W, H), borderValue=borde)
    nuevas = []
    for c, x1, y1, x2, y2 in cajas:
        p = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), M)[0]
        nuevas.append((c, float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())))
    f = out.astype(np.float32) * rng.uniform(0.75, 1.1) + rng.uniform(-15, 15)
    if rng.random() < 0.4:  # franjas tenues tipo moiré
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        f += 5 * np.sin(yy * rng.uniform(0.4, 1.5) + rng.uniform(0, 6))
    return np.clip(f, 0, 255).astype(np.uint8), nuevas


def degradar(img, rng):
    if rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.6))
    if rng.random() < 0.3:
        H, W = img.shape[:2]
        k = rng.uniform(0.45, 0.8)
        img = cv2.resize(cv2.resize(img, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.6:
        ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(35, 90))])
        img = cv2.imdecode(b, cv2.IMREAD_COLOR)
    return img


TAM = [(640, 412), (428, 487), (640, 360), (1365, 768), (1408, 768), (640, 640), (720, 960), (960, 720), (800, 600)]
TEXTOS = ["Fine crack", "Mold patch", "Dirty", "Stain", "Hole", "Dent", "Normal", "Clean egg", "Grieta", "Moho", "Sucio", "Sano"]


def elegir(banco, rng, p_dano=0.5):
    """Mitad con daño, mitad sanos; dentro de cada mitad se mezclan los de la colección del profesor, los reales y los sintéticos."""
    if rng.random() < p_dano:
        t = rng.random()
        if t < 0.62 or not (banco["sano_ia"] or banco["sano_real"]):
            return banco["dano_profe"][rng.integers(0, len(banco["dano_profe"]))]
        if t < 0.80 and banco["dano_real"]:
            return banco["dano_real"][rng.integers(0, len(banco["dano_real"]))]
        for _ in range(5):
            b = banco["sano_ia"] if (rng.random() < 0.5 or not banco["sano_real"]) else banco["sano_real"]
            m = ensuciar(b[rng.integers(0, len(b))], rng)
            if m is not None:
                return m
        return banco["dano_profe"][rng.integers(0, len(banco["dano_profe"]))]
    b = banco["sano_ia"] if (rng.random() < 0.6 or not banco["sano_real"]) else banco["sano_real"]
    return b[rng.integers(0, len(b))]


def componer(banco, rng):
    W, H = TAM[rng.integers(0, len(TAM))]
    if rng.random() < 0.3:
        W, H = int(W * rng.uniform(0.8, 1.3)), int(H * rng.uniform(0.8, 1.3))
    lienzo, blanco = fondo(H, W, rng)
    cajas, textos = [], []
    t = rng.random()
    sombra = blanco and rng.random() < 0.3
    solo_profe_blanco = not blanco  # los recortes del profesor no tienen silueta fina para fondos de color: se usan igual, con borde suavizado
    if t < 0.45:      # un huevo
        e = elegir(banco, rng)
        lado = min(H, W) * rng.uniform(0.3, 0.92)
        rgb, sil, _ = variar(e, rng, lado)
        h, w = sil.shape
        if h < H and w < W:
            x, y = int(rng.uniform(0, W - w)), int(rng.uniform(0, H - h))
            if rng.random() < 0.6:
                x, y = (W - w) // 2 + int(rng.normal(0, 0.05 * W)), (H - h) // 2 + int(rng.normal(0, 0.05 * H))
            if pegar(lienzo, rgb, sil, x, y, blanco, sombra, rng):
                cajas.append((e["clase"], *caja_de(sil, x, y)))
    elif t < 0.82:    # lámina en rejilla
        filas, cols = [(2, 5), (2, 5), (2, 4), (3, 4), (1, 3), (1, 5), (2, 3), (3, 5), (2, 6), (1, 2)][rng.integers(0, 10)]
        if H > W:
            filas, cols = cols, filas
        rotulos = rng.random() < 0.2
        cw, ch = W / cols, H / filas
        p = rng.uniform(0.3, 0.75)
        for r in range(filas):
            for c in range(cols):
                if rng.random() < 0.06:
                    continue
                e = elegir(banco, rng, p)
                lado = min(cw, ch) * rng.uniform(0.7, 0.93) * (0.85 if rotulos else 1)
                rgb, sil, _ = variar(e, rng, lado)
                h, w = sil.shape
                x = int(c * cw + (cw - w) / 2 + rng.normal(0, 0.02 * cw))
                y = int(r * ch + (ch - h) / 2 + rng.normal(0, 0.02 * ch) - (0.06 * ch if rotulos else 0))
                if pegar(lienzo, rgb, sil, x, y, blanco, sombra, rng):
                    cajas.append((e["clase"], *caja_de(sil, x, y)))
                    if rotulos:
                        txt = TEXTOS[rng.integers(0, len(TEXTOS))]
                        col = (30, 30, 30) if blanco else (235, 235, 235)
                        textos.append((txt, (int(c * cw + cw * 0.15), int(y + h + 0.07 * ch)), max(0.35, cw / 420), col))
    else:             # varios huevos sueltos
        ocup = []
        for _ in range(rng.integers(2, 7)):
            e = elegir(banco, rng)
            lado = min(H, W) * rng.uniform(0.18, 0.45)
            rgb, sil, _ = variar(e, rng, lado)
            h, w = sil.shape
            for _i in range(12):
                x, y = int(rng.uniform(0, max(1, W - w))), int(rng.uniform(0, max(1, H - h)))
                if all(x + w < a or a2 < x or y + h < b or b2 < y for a, b, a2, b2 in ocup):
                    if pegar(lienzo, rgb, sil, x, y, blanco, sombra, rng):
                        ocup.append((x, y, x + w, y + h))
                        cajas.append((e["clase"], *caja_de(sil, x, y)))
                    break
    img = np.clip(lienzo, 0, 255).astype(np.uint8)
    for txt, pos, esc, col in textos:
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, esc, col, 1, cv2.LINE_AA)
    if rng.random() < 0.25:
        img, cajas = pantalla(img, cajas, rng)
    img = degradar(img, rng)
    return img, cajas


def caja_de(sil, x, y):
    ys, xs = np.where(sil > 0.5)
    return x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1


def guardar_yolo(out, split, nombre, img, cajas):
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(out, split, sub), exist_ok=True)
    H, W = img.shape[:2]
    cv2.imwrite(os.path.join(out, split, "images", nombre + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(out, split, "labels", nombre + ".txt"), "w") as f:
        for c, x1, y1, x2, y2 in cajas:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 > 4 and y2 - y1 > 4:
                f.write(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")


def generar_detector(banco, originales, out, n, seed, copias_originales=3):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    for split, cuantos in n.items():
        for i in range(cuantos):
            img, cajas = componer(banco, rng)
            guardar_yolo(out, split, f"prf_{split}_{i:05d}", img, cajas)
    for k, o in enumerate(originales):  # las imágenes del profesor tal cual
        for r in range(copias_originales):
            guardar_yolo(out, "train", f"prf_orig_{k:03d}_r{r}", o["rgb"], o["cajas"])
        guardar_yolo(out, "originales", f"prf_orig_{k:03d}", o["rgb"], o["cajas"])


# ------------------------------------------------------------------ zona dañada
def recorte_seg(e, rng, aumentar=True):
    """Recorte como el de la app (caja del huevo + 10 %, estirado a 192x192) con silueta y zona dañada."""
    lado = rng.uniform(150, 260) if aumentar else 220
    if aumentar:
        rgb, sil, dano = variar(e, rng, lado)
    else:
        rgb, sil, dano = e["rgb"], e["sil"].astype(np.float32), e["dano"]
    h, w = sil.shape
    px, py = int(w * rng.uniform(0.06, 0.16)) if aumentar else int(0.1 * w), int(h * rng.uniform(0.06, 0.16)) if aumentar else int(0.1 * h)
    lienzo, blanco = fondo(h + 2 * py, w + 2 * px, rng) if aumentar else (np.full((h + 2 * py, w + 2 * px, 3), 255, np.float32), True)
    pegar(lienzo, rgb, sil, px, py, blanco, False, rng)
    img = np.clip(lienzo, 0, 255).astype(np.uint8)
    S = np.zeros(img.shape[:2], np.uint8)
    D = np.zeros(img.shape[:2], np.uint8)
    S[py:py + h, px:px + w] = (sil > 0.5) * 255
    D[py:py + h, px:px + w] = (dano & (sil > 0.5)) * 255
    if aumentar:
        img = degradar(img, rng)
    return (cv2.resize(img, (SEG, SEG), interpolation=cv2.INTER_LINEAR), cv2.resize(S, (SEG, SEG), interpolation=cv2.INTER_NEAREST),
            cv2.resize(D, (SEG, SEG), interpolation=cv2.INTER_NEAREST))


def generar_seg(banco, out, n, seed):
    """n recortes por split: con daño (profesor y manchas sintéticas) y sanos (máscara de daño vacía)."""
    rng = np.random.default_rng(seed)
    for split, cuantos in n.items():
        if not cuantos:
            continue
        d = os.path.join(out, split)
        os.makedirs(d, exist_ok=True)
        X, Y = [], []
        for i in range(cuantos):
            t = rng.random()
            if t < 0.45:
                e = banco["dano_profe"][rng.integers(0, len(banco["dano_profe"]))]
            elif t < 0.72:
                e = None
                while e is None:
                    b = banco["sano_ia"] if (rng.random() < 0.5 or not banco["sano_real"]) else banco["sano_real"]
                    e = ensuciar(b[rng.integers(0, len(b))], rng)
            else:
                b = banco["sano_ia"] if (rng.random() < 0.5 or not banco["sano_real"]) else banco["sano_real"]
                e = b[rng.integers(0, len(b))]
            img, S, D = recorte_seg(e, rng)
            X.append(img)
            Y.append(np.stack([S, D], -1))
        np.savez_compressed(os.path.join(d, "datos.npz"), x=np.stack(X), y=np.stack(Y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profe", required=True, help="carpeta imagenes/ del repositorio adiacla/huevos")
    ap.add_argument("--anot", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--reales", help="JSON [{file, clase, cajas, split}] de fotos reales con caja")
    ap.add_argument("--dano", help=".tflite del modelo de zona dañada (da la silueta de los huevos reales)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, nargs=3, default=[2600, 300, 300], metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--seg", type=int, nargs=2, default=[0, 0], metavar=("TRAIN", "VALID"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    reales = json.load(open(a.reales, encoding="utf-8")) if a.reales else []
    banco, originales = armar_banco(a.profe, a.anot, reales, a.dano, rng)
    if sum(a.n):
        generar_detector(banco, originales, a.out, dict(zip(("train", "valid", "test"), a.n)), a.seed + 1)
    if sum(a.seg):
        generar_seg(banco, os.path.join(a.out, "seg"), dict(zip(("train", "valid"), a.seg)), a.seed + 2)
    print("listo:", a.out)
