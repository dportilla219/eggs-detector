# Documentación del proyecto: detección de huevos con daños

Trabajo final de Ciencia de Datos (UNAB), Tema 1. Este documento cuenta de principio a fin cómo se hizo: datos, entrenamiento, evaluación, aplicación y despliegue. Para el detalle de cada parte:

| documento | contenido |
|---|---|
| [`README.md`](README.md) | Estado, modelos entregados, tablas de resultados e historial de versiones |
| [`MODELO_IO.md`](MODELO_IO.md) | Entrada, salida y decodificación de los dos modelos |
| [`app/README.md`](app/README.md) | La app web: pestañas, API, ejecución en local y despliegue |
| [`NOTAS.md`](NOTAS.md) | Notas de trabajo: decisiones y estado de cada versión |

## 1. Qué hace

Simula una línea de clasificación de huevos en una banda transportadora:

1. Una cámara ve uno o varios huevos.
2. Un **detector** ubica cada huevo y dice si está sano o con daño.
3. Un **segundo modelo** marca la zona dañada de cada huevo con daño y calcula qué porcentaje de la cáscara ocupa (la gravedad). Este es el diferencial del proyecto.
4. Según la gravedad, cada huevo va a una salida.

| salida | regla |
|---|---|
| Empaque | todos los huevos detectados están sanos |
| Industria | huevo con daño leve (menos del 15 % de la cáscara) |
| Descarte | huevo con daño del 15 % o más |
| Revisión manual | no se detectó ningún huevo |

Un clasificador binario descartaría todos los huevos con daño; con la zona dañada, los de daño leve se aprovechan.

## 2. Arquitectura

```
imagen ──► detector (YOLOv8n, 640×640) ──► cajas + clase (Crack / Intact)
                                               │
                              solo huevos Crack ▼
              recorte del huevo (192×192) ──► zona dañada (U-Net) ──► silueta + daño
                                                                        │
                                              gravedad = daño / silueta ▼
                                                        empaque · industria · descarte
```

| modelo | archivo | arquitectura | entrada | salida |
|---|---|---|---|---|
| Detector `v5` | `modelo/eggs_v5_fp32.tflite` (12,3 MB) | YOLOv8n | `[1,640,640,3]` RGB 0–1 | `[1,6,8400]`: caja + puntaje de `Crack` e `Intact` |
| Zona dañada `dano_v2` | `modelo/eggs_dano_v2_fp16.tflite` (4,9 MB) | U-Net con codificador MobileNetV2 (α = 0,5) | `[1,192,192,3]` RGB 0–1 | `[1,192,192,2]`: silueta del huevo y daño |

Reglas de decisión:
- Se descartan las detecciones con confianza menor de 0,5 y se aplica NMS con IoU 0,5.
- Un huevo es `Crack` solo si `puntaje_Intact <= 0,3 × puntaje_Crack`. Si no, es `Intact`.
- Gravedad: leve < 15 %, media < 35 %, grave ≥ 35 %.
- Si `dano_v2` no marca ninguna zona en un huevo `Crack`, se consulta `dano_v1`.

## 3. Datos

| fuente | uso |
|---|---|
| Dataset base en formato YOLOv8, dos clases, sin duplicados entre train, valid y test | Entrenamiento desde `v1` |
| Imágenes sintéticas: huevos intercambiados entre fondos (~2.600) | `v2`, para quitar el atajo del fondo |
| Dataset público [Egg-Defect-Detection](https://github.com/dakshkathuria346-gif/Egg-Defect-Detection) (fotos de celular), con cajas generadas con Grounding DINO | `v3` y `v4` |
| Fotos de Wikimedia Commons de huevos sanos y rajados | `v3`, `v4` y test independiente |
| Colección de daños del profesor ([adiacla/huevos](https://github.com/adiacla/huevos)): 147 huevos con agujeros, hundidos, cáscara rota, suciedad y moho, sueltos y en láminas | `v5` y `dano_v2`. Cajas en `v5/profe_huevos.json`, zonas en `v5/masks/` |
| Composiciones generadas con [`v5/datos.py`](v5/datos.py) (~3.000) | `v5` y `dano_v2` |
| Anotaciones propias de zona dañada: 420 huevos sobre una rejilla 10×10, con silueta de SAM 2.1 | `dano_v1` y `dano_v2` (`dano/anotaciones/`) |

**Problema del dataset base.** Todas las fotos de huevos sanos son de un mismo montaje (fondo gris, base negra) y todas las de otras fuentes son de huevos rajados. El primer modelo aprendió en parte a decidir por el fondo: marcaba como rajado 1 de cada 3 huevos sanos fuera del montaje. Casi todo el trabajo posterior consistió en quitar ese sesgo.

**Qué genera `v5/datos.py`.** Imágenes con uno o varios huevos, mezclando dañados y sanos; manchas, moho, barro, polvo y óxido sintéticos con su máscara exacta; sellos, fechas y logotipos impresos sobre huevos sanos (para que una marca no cuente como daño); efecto de pantalla y perspectiva (para cuando la cámara apunta a un monitor); y los recortes de 192×192 para el modelo de zona dañada.

## 4. Entrenamiento

Todo se entrenó en Google Colab con GPU T4. Cada versión tiene su notebook y nunca se sobrescribe un run anterior.

### Detector

Todas las versiones usan Ultralytics YOLOv8n, imágenes de 640 px y lotes de 16.

| versión | notebook | de dónde parte | qué cambió | resultado |
|---|---|---|---|---|
| `v1` | `eggs_train.ipynb` | `yolov8n.pt` (COCO), 100 épocas con parada temprana (paró en la 91) | Entrenamiento base | Aprende el atajo del fondo |
| `v2` | `eggs_v2.ipynb` | `v1`, 60 épocas | Imágenes sintéticas con huevos intercambiados entre fondos | Quita el atajo; falla en fotos reales de otras fuentes |
| `v3` | `eggs_v3.ipynb` | `v2` | Fotos reales de otras fuentes | Acierta los sanos reales, pierde rajados. No se entregó |
| `v4` | `eggs_v4.ipynb` | `v3`, 15 épocas | Mismo material con el split intercalado por subgrupo | Acierta fotos reales no vistas |
| `v5` | `eggs_v5.ipynb` y `v5/notebooks/` | `v4` | Daños de todo tipo y varios huevos por imagen | Modelo actual |

`v5` se hizo en tres pasos:
1. **Ronda 1** (20 épocas desde `v4`): colección del profesor y composiciones de `v5/datos.py`. Acertaba toda la colección, pero marcaba como daño huevos sanos con sello impreso.
2. **Ronda 2**, `v5b` (desde la ronda 1, parada temprana en la época 10): sellos y logotipos sobre huevos sanos, y menos manchas sintéticas.
3. **Promedio de pesos** de los dos checkpoints de la ronda 2 (`best.pt` y `last.pt`) y elección de la regla de decisión 0,3 con imágenes no vistas.

El resultado se exporta a TFLite FP32 con entrada NHWC. `v5` no tiene versión INT8 (la calibración agotaba la memoria de Colab).

### Zona dañada

| versión | notebook | entrenamiento |
|---|---|---|
| `dano_v1` | `eggs_dano.ipynb` | 80 épocas con las 420 anotaciones propias |
| `dano_v2` | `eggs_dano_v2.ipynb` | 80 épocas desde ImageNet; cada lote mezcla recortes originales y recortes de `v5/datos.py` (daños del profesor, manchas sintéticas y huevos sanos con máscara vacía) |

La pérdida es entropía cruzada binaria más Dice, con el doble de peso para el canal de daño. Se guarda el modelo con mejor IoU de daño en validación. Se eligió un segundo modelo en vez de YOLOv8-seg porque solo una parte pequeña de los huevos rajados tiene la zona anotada.

## 5. Evaluación

Las mediciones se hacen con la misma tubería de la app (`.tflite` en CPU, confianza 0,5), no solo con las métricas del entrenamiento.

| prueba | `v4` | `v5` |
|---|---|---|
| Colección del profesor, por huevo (vista al entrenar `v5`) | 20/147 | 147/147 |
| La misma, simulando la cámara apuntando a una pantalla | 77/588 | 584/588 |
| Imágenes no vistas, con daño | 1070/1146 (93,4 %) | 1088/1146 (94,9 %) |
| Imágenes no vistas, sanos | 416/441 (94,3 %) | 415/441 (94,1 %) |
| Test original, mAP50-95 (Crack / Intact) | — | 0,976 / 0,976 |

Zona dañada en el test original: `dano_v2` tiene IoU 0,63 y error de gravedad de ±7,5 puntos (`dano_v1`: 0,64 y ±9,6).

Los archivos con el detalle están en `resultados/` (uno por versión) y los scripts de evaluación en `v5/eval_*.py` y `app/server/evaluate.py`.

**Límites conocidos.**
- Las grietas muy finas o de espaldas a la cámara pueden pasar por sanas.
- Los huevos sanos con sello impreso todavía fallan a veces.
- En fotos de celular de otra fuente, `v5` baja un poco frente a `v4` (rajados 207/226 frente a 212/226).
- La colección del profesor se usó para entrenar, así que su 147/147 no mide generalización.
- `dano_v2` localiza menos grietas finas que `dano_v1` en fotos de celular (81 % frente a 98 %); por eso `dano_v1` queda de respaldo.

## 6. Aplicación web

Código en [`app/`](app/README.md).

| parte | archivos | qué hace |
|---|---|---|
| Servidor | `app/server/main.py` | API con FastAPI |
| Tubería | `app/server/pipeline.py` | Preprocesado, detector, NMS, recorte, zona dañada y gravedad, con LiteRT en CPU |
| Rutas | `app/server/routing.py` | Decide empaque, industria o descarte |
| Interfaz | `app/web/` | HTML, CSS y JavaScript sin frameworks |
| Sin servidor | `app/web/local.js`, `app/deploy/build_static.py` | Los mismos `.tflite` ejecutados en el navegador, como respaldo |

Pestañas de la interfaz:
- **Banda transportadora**: pasa imágenes del test por la cámara simulada y muestra aciertos, matriz de confusión, latencia y huevos por minuto.
- **Analizar imagen**: subir o tomar una foto; dibuja cajas, zona dañada, puntajes y gravedad.
- **Cámara / video**: envía fotogramas de 640 px al servidor y decide por mayoría en los últimos fotogramas. Con varios huevos a la vez informa de cada uno.
- **Modelo y métricas**: arquitectura y resultados.

Rutas principales de la API: `GET /api/health`, `GET /api/info`, `GET /api/samples`, `POST /api/predict` (una imagen) y `POST /api/samples/{id}/predict`.

Para ejecutarla en local:

```bash
python -m venv .venv && source .venv/bin/activate      # Linux / macOS
pip install -r app/requirements.txt
cd app/server
EGGS_SAMPLES_DIR=/ruta/a/eggs_v2/test uvicorn main:app --port 8000
```

## 7. Despliegue

La app corre en una instancia EC2 de AWS con Ubuntu 24.04 (laboratorio de AWS Academy).

```
celular / PC ──HTTPS 443──► Caddy ──► uvicorn (FastAPI, puerto 8000) ──► LiteRT en CPU
```

- **Instalación.** `app/deploy/install.sh` crea el entorno virtual, instala las dependencias, evalúa el test con la tubería completa, registra el servicio de systemd y configura Caddy.
- **Servicio.** `app/deploy/eggs-detector.service` arranca la app al encender la instancia. Los modelos se eligen ahí con variables de entorno.
- **HTTPS.** Caddy obtiene el certificado de Let's Encrypt para el nombre `<ip-con-guiones>.sslip.io`, sin comprar dominio. El celular solo permite usar la cámara en vivo por HTTPS.
- **Cambio de IP.** El laboratorio se apaga solo a las pocas horas y al encenderlo cambia la IP pública. `app/deploy/caddy-host.sh` ajusta el nombre en cada arranque.
- **Grupo de seguridad.** TCP 80 y 443 (HTTPS) y 8000 (HTTP directo).

Primera instalación en la instancia:

```bash
git clone https://github.com/dportilla219/eggs-detector.git ~/eggs-detector
bash ~/eggs-detector/app/deploy/install.sh
```

Para actualizar tras un cambio en el repositorio:

```bash
cd ~/eggs-detector && git pull && bash app/deploy/install.sh
```

Variables del servicio:

| variable | valor actual | para volver a la versión anterior |
|---|---|---|
| `EGGS_DET_FILE` | `eggs_v5_fp32.tflite` | `eggs_v4_fp32.tflite` |
| `EGGS_SEG_FILE` | `eggs_dano_v2_fp16.tflite` | `eggs_dano_v1_fp16.tflite` |
| `EGGS_SEG_FALLBACK` | `eggs_dano_v1_fp16.tflite` | quitar la línea |
| `EGGS_CRACK_RATIO` | `0.3` | quitar la línea |

## 8. Cómo repetir el entrenamiento

1. Subir los cuatro zips del dataset (`eggs_v2_parte*.zip`) a Google Drive, en `MyDrive/eggs_v2/`.
2. Abrir el notebook de la versión en Colab con GPU y usar *Ejecutar todo*. Las celdas que ya hicieron su trabajo se saltan.
3. Los resultados quedan en Drive (`runs/<versión>` y `exports/<versión>`); los `.tflite` se copian a `modelo/`.

Los notebooks de `v5/notebooks/` y `eggs_dano_v2.ipynb` leen el dataset y los pesos de una rama temporal del repositorio (`v5-tmp`) que ya no existe. Para repetirlos hay que volver a subir ahí los zips y los `.pt`.

## 9. Estructura del repositorio

| ruta | contenido |
|---|---|
| `modelo/` | Los `.tflite` de todas las versiones y `eggs_v5.pt` |
| `resultados/` | Métricas y verificaciones de cada versión |
| `eggs_*.ipynb` | Un notebook por versión del detector y de la zona dañada |
| `dano/` | Scripts y anotaciones del modelo de zona dañada |
| `v5/` | Generador de datos, etiquetas y máscaras de la colección del profesor, notebooks de la ronda 2 y scripts de evaluación |
| `app/` | Servidor, interfaz web y archivos de despliegue |
