# App web: detección de huevos con daños y simulación de banda

Despliegue de los modelos de este repositorio como aplicación web, con inferencia en el servidor.

- **`v2`** (`modelo/eggs_v2_fp32.tflite`, YOLOv8n): encuentra cada huevo y lo clasifica como `Crack` o `Intact`.
- **`dano_v1`** (`modelo/eggs_dano_v1_fp16.tflite`, U-Net MobileNetV2): solo sobre los huevos `Crack`, marca la **zona dañada** y calcula la gravedad. Es el diferencial del proyecto.

La app sigue al pie de la letra la entrada y la salida descritas en [`MODELO_IO.md`](../MODELO_IO.md). Solo cambia el preprocesado del detector: por defecto usa *letterbox* (la imagen completa con bandas grises, como en el entrenamiento de Ultralytics) en lugar del cuadrado centrado de la app móvil, para no recortar las fotos. El modo de cuadrado centrado se puede elegir en la interfaz.

**Desplegada en AWS EC2:**
- **https://34-225-169-137.sslip.io** (HTTPS: funciona en cualquier celular o PC, incluida la cámara en vivo)
- http://34.225.169.137:8000 (HTTP directo; la cámara en vivo no funciona por HTTP, pero "Tomar foto" sí)

No hace falta Expo Go ni instalar nada: se abre en el navegador. Si la instancia cambia de IP pública (por ejemplo, al reiniciar un laboratorio de AWS Academy sin IP elástica), la dirección HTTPS pasa a ser `https://<ip-con-guiones>.sslip.io` automáticamente.

## Qué incluye

| pestaña | qué hace |
|---|---|
| **Banda transportadora** | Simula una línea de clasificación en tiempo real. Cada huevo (imagen del test real) pasa por la cámara, el servidor ejecuta `v2` y, si está rajado, `dano_v1`. Un desviador lo manda a su salida. Muestra el acierto frente a la etiqueta real, la matriz de confusión, la latencia y los huevos por minuto. |
| **Analizar imagen** | Subir una foto o elegir una del test. Dibuja las cajas y la zona dañada, y muestra los scores, la gravedad y el recorte de 192×192 que ve `dano_v1`. |
| **Cámara / video** | Envía fotogramas de la cámara o de un archivo de video (~4 por segundo) y suaviza la decisión por mayoría en los últimos 7. |
| **Modelo y métricas** | Arquitectura, métricas del equipo (`resultados/`), evaluación hecha por la propia app en el servidor y reglas de la banda. |

### Reglas de la banda (el diferencial en acción)

| ruta | regla |
|---|---|
| Empaque | todos los huevos detectados son `Intact` |
| Industria | `Crack` con gravedad < 15 % (leve): reproceso industrial |
| Descarte | `Crack` con gravedad ≥ 15 % (media o grave) |
| Revisión manual | no se detectó ningún huevo |

Un detector binario descartaría todos los rajados. Con la zona dañada, los que tienen una grieta pequeña se aprovechan. El umbral del 15 % es el nivel "leve" de `MODELO_IO.md` y se cambia en `server/routing.py`.

## Evaluación en el servidor

`server/evaluate.py` recorre el split de test (382 imágenes) con la tubería completa. En la instancia EC2 (2 vCPU, CPU) da:

| grupo | n | acierto |
|---|---|---|
| Crack del montaje | 47 | 0.851 |
| Intact del montaje | 84 | 0.976 |
| Crack de otras fuentes | 251 | 1.000 |
| **Global** | **382** | **0.976** |

Coincide con `resultados/v2/resumen.json`, lo que confirma que el preprocesado y la decodificación del servidor son correctos. Latencia: ~160 ms por imagen (p50) con los dos modelos.

## Estructura

```
app/
├── server/
│   ├── main.py        API FastAPI + archivos estáticos
│   ├── pipeline.py    v2 → NMS → recorte +10 % → dano_v1 → gravedad
│   ├── routing.py     reglas de la banda
│   ├── samples.py     catálogo de imágenes de test (split YOLO) con su etiqueta real
│   └── evaluate.py    evaluación de la tubería completa
├── web/               interfaz (HTML + CSS + JS, sin compilación)
├── deploy/
│   ├── eggs-detector.service   servicio systemd (puerto 8000)
│   ├── Caddyfile               HTTPS con certificado automático (Let's Encrypt + sslip.io)
│   ├── caddy-host.sh           ajusta el nombre HTTPS si cambia la IP pública
│   └── install.sh              instalación en Ubuntu
└── requirements.txt
```

## Robustez

Probado contra la app desplegada (`/api/predict`):

| caso | resultado |
|---|---|
| JPG, PNG, WEBP, GIF, **HEIC de iPhone**, CMYK, escala de grises, PNG de 16 bits o con transparencia | se analizan (todo se convierte a RGB y se endereza según el EXIF) |
| foto de 8000×6000 px | se decodifica ya reducida (2048 px) sin agotar la RAM |
| archivo que no es imagen, vacío, de más de 15 MB, de menos de 32 px o con más de 80 MP | mensaje de error claro en español, sin caerse |
| 16 peticiones a la vez | 16/16 correctas; como mucho 3 imágenes en proceso a la vez y el resto espera turno |
| reinicio del servicio | `systemd` lo levanta solo; la página reintenta la conexión y avisa si el servidor no responde |

En el celular, "Tomar foto" reduce la imagen a 1600 px antes de subirla.

## API

| método | ruta | descripción |
|---|---|---|
| `POST` | `/api/predict` | `multipart/form-data`: `file` (imagen), `conf` (0.5), `mode` (`letterbox` \| `center`) |
| `POST` | `/api/samples/{id}/predict` | igual, con una imagen del test del servidor |
| `GET` | `/api/samples` | imágenes de test con su etiqueta real |
| `GET` | `/api/info` | modelos, métricas y reglas |
| `GET` | `/docs` | documentación interactiva (Swagger) |

```bash
curl -F "file=@huevo.jpg" http://<ip>:8000/api/predict
```

Cada huevo de la respuesta trae `box` (píxeles), `label`, `conf`, `scores` y, si es `Crack`, `damage` con `severity`, `level`, `rect` y la máscara en PNG (`mask_png`). La respuesta incluye `route` con la ruta decidida.

## Ejecutar en local

```bash
python -m venv .venv && source .venv/bin/activate      # Linux / macOS
pip install -r app/requirements.txt
cd app/server
EGGS_SAMPLES_DIR=/ruta/a/eggs_v2/test uvicorn main:app --port 8000
```

`ai-edge-litert` (el runtime oficial de TFLite) publica ruedas para Linux, macOS y Windows.

## Despliegue en EC2 (Ubuntu 24.04)

```bash
git clone https://github.com/dportilla219/eggs-detector.git ~/eggs-detector
# split de test (images/ + labels/) de eggs_v2_parte4_valid_test.zip en ~/eggs-data/test
bash ~/eggs-detector/app/deploy/install.sh
```

Después hay que abrir en el grupo de seguridad los puertos **8000** (HTTP directo), **80** y **443** (HTTPS con Caddy). Los dos servicios arrancan solos al reiniciar la instancia (`systemctl status eggs-detector caddy`).

El dataset no se sube al repositorio, igual que en el resto del proyecto. Sin `EGGS_SAMPLES_DIR`, la app funciona igual, pero la banda y la galería quedan vacías.

Para cambiar de detector (por ejemplo a `v3`) basta con cambiar `EGGS_DET_FILE` en `deploy/eggs-detector.service`: todos tienen la misma entrada y salida.

**Cámara:** los navegadores solo dan acceso a la cámara en páginas HTTPS. Por eso la app se sirve también por HTTPS; desde la dirección HTTP, la pestaña de cámara ofrece un botón para abrir la versión segura.
