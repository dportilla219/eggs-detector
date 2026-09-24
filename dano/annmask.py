import re, cv2, numpy as np
G = 10  # rejilla 10x10 sobre el recorte del huevo: filas A-J, columnas 0-9


def parse(line):
    """'12: B3-5 C4' -> (12, {(1,3),(1,4),(1,5),(2,4)})"""
    i, rest = line.split(':', 1); cells = set()
    for tok in rest.split():
        m = re.fullmatch(r'([A-J])(\d)(?:-(\d))?', tok)
        if not m: continue
        r = 'ABCDEFGHIJ'.index(m[1]); a = int(m[2]); b = int(m[3]) if m[3] else a
        for c in range(a, b + 1): cells.add((r, c))
    return int(i.strip().lstrip('#')), cells


def cellmask(shape, cells):
    h, w = shape; m = np.zeros((h, w), np.uint8)
    for r, c in cells: m[int(r * h / G):int((r + 1) * h / G), int(c * w / G):int((c + 1) * w / G)] = 1
    return m


def refine(crop, cells, eggcrop=None):
    """zona dañada = celdas marcadas suavizadas (contorno redondeado) ∩ silueta del huevo"""
    h, w = crop.shape[:2]; cm = cellmask((h, w), cells).astype(np.float32)
    if not cells: return cm.astype(np.uint8)
    out = (cv2.GaussianBlur(cm, (0, 0), 0.35 * min(h, w) / G) > 0.5).astype(np.uint8)
    if eggcrop is not None: out &= eggcrop.astype(np.uint8)
    return out
