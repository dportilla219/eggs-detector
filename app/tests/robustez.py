"""Pruebas de robustez contra la app desplegada.

Uso: python robustez.py https://34-225-169-137.sslip.io foto_de_un_huevo.jpg
Requiere: pip install requests pillow pillow-heif numpy
"""
import io, sys, time, concurrent.futures as cf
import numpy as np, requests
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
BASE = sys.argv[1]
egg = Image.open(sys.argv[2]).convert("RGB")  # cualquier foto con un huevo

def enc(img, fmt, **kw):
    b = io.BytesIO(); img.save(b, format=fmt, **kw); return b.getvalue()

big = egg.resize((8000, 6000))
cases = {
    "jpeg normal": (enc(egg, "JPEG"), {}),
    "jpeg 8000x6000": (enc(big, "JPEG", quality=85), {}),
    "png 16 bits": (enc(Image.fromarray((np.asarray(egg.convert("L")).astype(np.uint16) * 257), "I;16"), "PNG"), {}),
    "png transparente": (enc(egg.convert("RGBA"), "PNG"), {}),
    "heic (iPhone)": (enc(egg, "HEIF"), {}),
    "webp": (enc(egg, "WEBP"), {}),
    "gif": (enc(egg.convert("P"), "GIF"), {}),
    "imagen 5x5": (enc(egg.resize((5, 5)), "PNG"), {}),
    "panorama 4000x60": (enc(egg.resize((4000, 60)), "JPEG"), {}),
    "gris (L)": (enc(egg.convert("L"), "JPEG"), {}),
    "CMYK": (enc(egg.convert("CMYK"), "JPEG"), {}),
    "no es imagen": (b"hola, esto es texto", {}),
    "archivo vacio": (b"", {}),
    "bomba 20000x20000": (enc(Image.new("L", (20000, 20000)), "PNG"), {}),
    "16 MB": (b"\xff\xd8" + b"0" * (16 * 1024 * 1024), {}),
    "conf=2": (enc(egg, "JPEG"), {"conf": "2"}),
    "conf=abc": (enc(egg, "JPEG"), {"conf": "abc"}),
    "mode=raro": (enc(egg, "JPEG"), {"mode": "raro"}),
    "return_image": (enc(egg, "JPEG"), {"return_image": "true"}),
}
for name, (data, form) in cases.items():
    t = time.time()
    try:
        r = requests.post(f"{BASE}/api/predict", files={"file": ("x", data)}, data=form, timeout=90)
        j = r.json()
        info = f"huevos={len(j['eggs'])} ruta={j['route']} {j['width']}x{j['height']}" + (" +imagen" if "image" in j else "") if r.ok else j.get("detail")
    except Exception as e:
        r, info = None, f"EXCEPCION {e}"
    print(f"{name:20s} -> {getattr(r, 'status_code', '---')} {time.time() - t:5.2f}s  {info}")

for path in ("/api/samples/999999/predict", "/api/samples/-1/image"):
    r = requests.request("POST" if "predict" in path else "GET", BASE + path, timeout=30)
    print(f"{path:30s} -> {r.status_code} {r.json().get('detail')}")

# 16 peticiones a la vez (varias personas usando la app en la sustentación)
data = enc(egg, "JPEG")
def one(i):
    t = time.time(); r = requests.post(f"{BASE}/api/predict", files={"file": ("x.jpg", data)}, timeout=120); return r.status_code, time.time() - t
t0 = time.time()
with cf.ThreadPoolExecutor(16) as ex:
    res = list(ex.map(one, range(16)))
codes = [c for c, _ in res]; ts = sorted(t for _, t in res)
print(f"16 simultaneas: codigos={sorted(set(codes))} ok={codes.count(200)}/16 tiempo max={ts[-1]:.1f}s mediana={ts[8]:.1f}s total={time.time()-t0:.1f}s")
