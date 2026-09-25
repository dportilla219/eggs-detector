"""Catálogo de imágenes de prueba (split YOLO: images/ + labels/) con su etiqueta real."""

from __future__ import annotations

import os

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_samples(split_dir: str | None) -> list[dict]:
    if not split_dir or not os.path.isdir(os.path.join(split_dir, "images")):
        return []
    items = []
    img_dir, lab_dir = os.path.join(split_dir, "images"), os.path.join(split_dir, "labels")
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(name)[0]
        objs: list[tuple[int, list[float]]] = []
        lab = os.path.join(lab_dir, stem + ".txt")
        if os.path.exists(lab):
            with open(lab) as fh:
                for line in fh:
                    p = line.split()
                    if len(p) >= 5:
                        objs.append((int(float(p[0])), [float(v) for v in p[1:5]]))
        if not objs:
            continue
        with Image.open(os.path.join(img_dir, name)) as im:  # solo lee la cabecera
            w, h = im.size
        classes = [c for c, _ in objs]
        gt = 0 if 0 in classes else 1  # la imagen es "Crack" si tiene algún huevo rajado
        # caja (cx, cy, w, h normalizados) del huevo más grande, para la miniatura de la banda
        box = max(objs, key=lambda o: o[1][2] * o[1][3])[1]
        items.append({
            "id": len(items),
            "file": name,
            "w": w,
            "h": h,
            "gt": gt,
            "gt_label": "Crack" if gt == 0 else "Intact",
            "n_eggs": len(objs),
            "box": [round(v, 4) for v in box],
            # las fotos del montaje (fondo gris, 224 px) empiezan por ec_egg
            "source": "montaje" if stem.startswith("ec_egg") else "otras",
        })
    return items
