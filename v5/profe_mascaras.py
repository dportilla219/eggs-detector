"""v5: cajas finales de los huevos del profesor + máscara de silueta y máscara de daño automática
(dano_v1 ∪ anomalía de color respecto del sombreado liso del huevo). Guarda hojas para revisar."""
import json, os, sys
import cv2, numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(R, "repo_profe", "imagenes")
OUT = os.path.join(R, "v5", "profe")
C = json.load(open(os.path.join(OUT, "cajas_auto.json"), encoding="utf-8"))
MANUAL = json.load(open(os.path.join(OUT, "manual.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "manual.json")) else {}
it = Interpreter(model_path=os.path.join(R, "repo", "modelo", "eggs_dano_v1_fp16.tflite")); it.allocate_tensors()
ii, oo = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
S, PAD = 192, 0.10


def leer(p):
    im = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    return np.asarray(bg.convert("RGB"))


def silueta(crop):
    """Huevo sobre fondo blanco: casco convexo de lo no-blanco; si el huevo es blanco y se pierde, elipse de la caja."""
    g = crop.min(axis=2)
    m = (g < 236).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    full = np.zeros_like(m)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(full, [cv2.convexHull(np.vstack(cs))], -1, 1, -1)
    if full.sum() < 0.70 * h * w:
        full = np.zeros_like(m); cv2.ellipse(full, (w // 2, h // 2), (int(w * 0.49), int(h * 0.49)), 0, 0, 360, 1, -1)
    return full.astype(bool)


def anomalia(crop, sil, k):
    """Lo que se sale del huevo liso: color distinto del dominante, o luminosidad lejos del perfil por anillos
    (mediana de L por anillo, forzada a no bajar hacia el centro, para que una mancha central no se tome por sombra)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = sil.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    inner = cv2.erode(sil.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    # color dominante: mediana en el huevo
    a0, b0 = np.median(A[inner]), np.median(B[inner])
    dc = np.sqrt((A - a0) ** 2 + (B - b0) ** 2)
    nb = 12
    idx = np.clip((r * nb).astype(int), 0, nb - 1)
    med = np.array([np.percentile(L[inner & (idx == i)], 60) if (inner & (idx == i)).sum() > 20 else np.nan for i in range(nb)])
    for i in range(nb - 2, -1, -1):  # de fuera hacia dentro: no puede oscurecerse
        if np.isnan(med[i + 1]): continue
        if np.isnan(med[i]) or med[i] < med[i + 1]: med[i] = med[i + 1]
    med = np.where(np.isnan(med), np.nanmax(med), med)
    Lexp = np.interp(r * nb - 0.5, np.arange(nb), med)
    dl = L - Lexp
    # degradado de luz (arriba claro, abajo sombra): plano ajustado al residuo con recorte de extremos
    sel = inner.copy()
    P = np.stack([np.ones_like(r), (xx - w / 2) / w, (yy - h / 2) / h], -1).astype(np.float32)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(P[sel], dl[sel], rcond=None)
        res = dl - P @ coef
        lo, hi = np.percentile(res[inner], [20, 85])
        sel = inner & (res > lo) & (res < hi)
    dl = dl - P @ coef
    sc = float(np.clip(np.median(dc[inner]) * 1.4826, 2.5, 4.0))
    sl = float(np.clip(np.median(np.abs(dl[inner])) * 1.4826, 2.0, 4.0))
    m = inner & ((dc > k * 2.0 * sc) | (dl < -k * 2.2 * sl) | (dl > k * 2.6 * sl))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        if st[i, 4] < 0.003 * sil.sum(): m[lb == i] = 0
    return m.astype(bool)


def dano_v1(img, box):
    x1, y1, x2, y2 = box; bw, bh = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    b = [int(max(0, x1 - PAD * bw)), int(max(0, y1 - PAD * bh)), int(min(W, x2 + PAD * bw)), int(min(H, y2 + PAD * bh))]
    crop = cv2.resize(img[b[1]:b[3], b[0]:b[2]], (S, S), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    it.set_tensor(ii, crop[None]); it.invoke(); o = it.get_tensor(oo)[0]
    d = ((o[..., 1] > 0.5) & (o[..., 0] > 0.5)).astype(np.uint8)
    d = cv2.resize(d, (b[2] - b[0], b[3] - b[1]), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((H, W), np.uint8); full[b[1]:b[3], b[0]:b[2]] = d
    return full[y1:y2, x1:x2].astype(bool)


K = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
FORZAR = sys.argv[2] if len(sys.argv) > 2 else None
final, celdas = {}, []
os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
gid = 0
for f, d in C.items():
    img = leer(os.path.join(SRC, f))
    eggs = []
    for k, box in enumerate(d["cajas"]):
        x1, y1, x2, y2 = box
        if (y2 - y1) < 80:  # rótulos de texto
            continue
        crop = img[y1:y2, x1:x2]
        man = MANUAL.get(str(gid), {})
        clase = man.get("clase", 1 if f.startswith("sano") else 0)
        sil = silueta(crop)
        modo = FORZAR or man.get("modo", "union")
        a, v = anomalia(crop, sil, man.get("k", K)), dano_v1(img, box) & sil
        if clase == 1 or modo in ("nada", "elipse"): m = np.zeros_like(sil)
        elif modo == "v1": m = v
        elif modo in ("anom", "anom_elipse"): m = a
        elif modo == "todo": m = cv2.erode(sil.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        else: m = v
        dentro = np.zeros(sil.shape, bool)
        for (ex, ey, erx, ery, val) in man.get("elipses", []):  # coordenadas relativas a la caja; val 1 añade, 0 borra
            e = np.zeros(sil.shape, np.uint8); h, w = sil.shape
            cv2.ellipse(e, (int(ex * w), int(ey * h)), (int(erx * w), int(ery * h)), 0, 0, 360, 1, -1)
            if modo == "anom_elipse": dentro |= e.astype(bool)
            else: m = (m | e.astype(bool)) if val else (m & ~e.astype(bool))
        if modo == "anom_elipse": m &= dentro
        m &= sil
        cv2.imwrite(os.path.join(OUT, "masks", f"{gid:03d}_sil.png"), sil.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(OUT, "masks", f"{gid:03d}_dano.png"), m.astype(np.uint8) * 255)
        borde = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.erode(sil.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        eggs.append({"id": gid, "caja": box, "clase": clase, "omitir": bool(man.get("omitir", False)), "sucio": bool(man.get("sucio", False)),
                     "dano_frac": round(float(m.sum() / max(1, sil.sum())), 4), "toca_borde": bool(borde.sum() > 0.01 * sil.sum())})
        T = 176
        a0 = cv2.resize(crop, (T, T)); ov = crop.copy(); ov[m] = (0.45 * ov[m] + 0.55 * np.array([230, 40, 160])).astype(np.uint8)
        cs, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov, cs, -1, (0, 160, 255), 1)
        cell = np.full((T + 16, 2 * T, 3), 25, np.uint8); cell[:T, :T] = a0; cell[:T, T:] = cv2.resize(ov, (T, T))
        cv2.putText(cell, f"{gid} {m.sum() / max(1, sil.sum()):.0%}{' OMITIR' if man.get('omitir') else ''}{' SANO' if clase == 1 else (' sucio' if man.get('sucio') else '')}", (3, T + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        celdas.append(cell); gid += 1
    final[f] = {"W": d["W"], "H": d["H"], "huevos": eggs}
json.dump(final, open(os.path.join(OUT, "profe_huevos.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
cols = 5
for s0 in range(0, len(celdas), 30):
    part = celdas[s0:s0 + 30]; rows = (len(part) + cols - 1) // cols
    sh = np.full((rows * (176 + 16), cols * 352, 3), 25, np.uint8)
    for i, c in enumerate(part): sh[(i // cols) * 192:(i // cols) * 192 + 192, (i % cols) * 352:(i % cols) * 352 + 352] = c
    cv2.imwrite(os.path.join(OUT, f"mask_{s0 // 30}.jpg"), cv2.cvtColor(sh, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(gid, "huevos")
