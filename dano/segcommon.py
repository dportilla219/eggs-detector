# Utilidades compartidas: recorte del huevo (idéntico al que hará la app) y aumentos
import glob, os, cv2, numpy as np

SEG_DIR = os.environ.get('SEG_DIR', 'seg')   # dataset de recortes generado por build.py

S = 192        # lado de la entrada del segmentador
PAD = 0.10     # la caja se expande un 10 % (5 % por lado) antes de recortar


def load_split(split):
    items = []
    for f in sorted(glob.glob(f'{SEG_DIR}/{split}/*.npz')):
        d = np.load(f)
        im = cv2.cvtColor(cv2.imdecode(d['img'], 1), cv2.COLOR_BGR2RGB)
        items.append(dict(name=f, img=im, box=d['box'], egg=d['egg'], dmg=d['dmg'], pos=f.endswith('_p.npz')))
    return items


def crop(it, jitter=None, rng=None):
    """Recorta la caja expandida y la estira a S×S (lo mismo que crop+scale del resize-plugin)."""
    x1, y1, x2, y2 = it['box']; w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if jitter:
        cx += rng.uniform(-jitter, jitter) * w; cy += rng.uniform(-jitter, jitter) * h
        w *= rng.uniform(1 - jitter, 1 + jitter); h *= rng.uniform(1 - jitter, 1 + jitter)
    w *= 1 + PAD; h *= 1 + PAD
    src = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]])
    dst = np.float32([[0, 0], [S, 0], [0, S]])
    M = cv2.getAffineTransform(src, dst)
    im = cv2.warpAffine(it['img'], M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = np.stack([it['egg'], it['dmg']], -1).astype(np.uint8) * 255
    m = cv2.warpAffine(m, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
    return im, (m > 127).astype(np.float32)


def augment(im, m, rng):
    k = rng.integers(4); im, m = np.rot90(im, k), np.rot90(m, k)
    if rng.random() < .5: im, m = im[:, ::-1], m[:, ::-1]
    im = np.ascontiguousarray(im).astype(np.float32); m = np.ascontiguousarray(m)
    # color: brillo, contraste, saturación, tono, balance de blancos
    hsv = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(.6, 1.4)
    im = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    im = (im - im.mean()) * rng.uniform(.7, 1.3) + im.mean() + rng.uniform(-35, 35)
    im *= rng.uniform(.9, 1.1, size=3)
    # cámara: baja resolución, desenfoque, ruido, JPEG
    if rng.random() < .4:
        r = rng.uniform(.3, .8); sm = cv2.resize(im, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        im = cv2.resize(sm, (S, S), interpolation=cv2.INTER_LINEAR)
    if rng.random() < .3: im = cv2.GaussianBlur(im, (0, 0), rng.uniform(.5, 1.5))
    if rng.random() < .4: im += rng.normal(0, rng.uniform(2, 8), im.shape)
    im = np.clip(im, 0, 255).astype(np.uint8)
    if rng.random() < .4:
        im = cv2.imdecode(cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])[1], 1)
    return im, m
