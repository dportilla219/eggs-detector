# Proyecto: detector de huevos rajados (YOLOv8 → TFLite)

## Objetivo
- Entrenar un modelo **YOLOv8 de detección** de huevos que funcione en **video en vivo** (frame a frame, sin tomar fotos).
- Clases: `0 = Crack` (huevo rajado), `1 = Intact` (huevo sano).

## Alcance y entregable
- Mi parte es **solo el modelo**. Mis compañeros lo integran en una app **React Native / Expo** (Android e iOS) con `react-native-vision-camera` + `react-native-fast-tflite`.
- **La app corre en un Expo development build (`expo-dev-client`), no en Expo Go.** vision-camera y fast-tflite son módulos nativos que Expo Go no incluye. Se consideró ejecutar en Expo Go (WebView + ONNX Runtime Web, o un servidor) y el equipo eligió el development build.
- Entregable final: un **`.tflite` ligero** (apto para tiempo real en el móvil) + **documentación de entrada/salida** (tamaño y formato de entrada, normalización, dtype, forma del tensor de salida y cómo decodificarlo: cajas, scores, clases, NMS).

## Entorno
- VS Code con un notebook `.ipynb` conectado a un **kernel de Google Colab con GPU**.
- Datos y resultados se guardan en **Google Drive**: `/content/drive/MyDrive/eggs_v2`, para no perderlos si Colab se desconecta. Los runs de entrenamiento deben escribir ahí (`project=` apuntando a Drive).

## Dataset
- Formato YOLOv8, en 4 zips: `eggs_v2_parte1_train.zip`, `eggs_v2_parte2_train.zip`, `eggs_v2_parte3_train.zip`, `eggs_v2_parte4_valid_test.zip` (copia local en `C:\Users\Usuario\Downloads\eggs_v2`; no se sube al repo). El original está en Drive: `/content/drive/MyDrive/eggs_v2`.
- Ya está limpio y **sin duplicados entre splits**.
- Desbalance: hay **~3,5× más Crack que Intact**.
- Sesgo de dominio: todas las imágenes de **Intact** son del mismo montaje (fondo gris, base negra, 224×224). Riesgo de que el modelo aprenda el fondo en vez del huevo; tenerlo en cuenta en aumentos de datos y en la evaluación.

- Fuente de cada imagen: los nombres que empiezan por `ec_egg` son del montaje. Train: Crack montaje 343, Crack otras fuentes 1767, Intact montaje 614, **Intact de otras fuentes 0**. Es decir, fondo = clase: el atajo es real y `v1` lo tiene disponible. `eggs_v2.ipynb` lo ataca con imágenes sintéticas (intercambio de huevos entre fondos).

## Notas técnicas
- Ultralytics 8.4: `best.pt` se elige solo por mAP50-95. El formato `tflite` se reemplazó por `litert` (litert-torch, solo Linux/macOS), que por defecto exporta la entrada en **NCHW**. `eggs_v2.ipynb` intercepta `torch2litert` para exportar en **NHWC** `[1,640,640,3]` (lo que da vision-camera-resize-plugin). Salida `[1,6,8400]`: cx,cy,w,h normalizados 0–1 + 2 scores con sigmoide, sin NMS.
- VS Code no recarga un `.ipynb` modificado en disco si está abierto y lo sobrescribe al guardar: tras editar un notebook, cerrar y volver a abrir la pestaña.

## Estado actual (2026-09-24)
- **Modelo entregado: `v2`** (fine-tune de `v1` con imágenes sintéticas, 60 épocas). Pesos en Drive: `runs/v2/weights/best.pt`. Exportado en `exports/v2/` y en `modelo/` del repo: `eggs_v2_fp32.tflite` (recomendado) y `eggs_v2_int8.tflite`. Umbral 0.5.
- Métricas: test original mAP50-95 0.970 (Crack) / 0.970 (Intact). Huevo sano fuera del montaje (sintético): `v1` 0.68 → `v2` 0.99. Punto débil: Crack del montaje, 0.85 (grietas poco visibles a 224 px). Detalle en `resultados/v2/resumen.json`.
- Los `.tflite` se verificaron localmente con la decodificación de `MODELO_IO.md` (18/18 correctos).
- Documentación para el equipo de la app: `MODELO_IO.md` y la sección "Guía para el equipo de la app" del README (pasos del development build y prompt de contexto para su IA).
- **Extra entregado: `dano_v1`** (factor diferencial): segundo modelo (MobileNetV2-0.5 + U-Net, Keras) que recibe el recorte de cada huevo Crack detectado por `v2` (caja +10 %, estirada a 192×192) y devuelve `[1,192,192,2]` = silueta del huevo + zona dañada → la app pinta el daño y calcula la gravedad (leve <15 %, media <35 %, grave). Anotaciones propias por rejilla 10×10 (`dano/anotaciones/`, 420 huevos) + silueta SAM 2.1. Notebook `eggs_dano.ipynb` (Colab GPU; clona este repo y usa `dano/*.py`). Se entrenó con el nombre de run `v3_dano` y después se renombró a `dano_v1` para que no parezca un sustituto de `v2`. En Drive sigue como `runs/v3_dano` y `exports/v3_dano/` (archivos `eggs_dano_v3_*`); en el repo, `modelo/eggs_dano_v1_fp16.tflite` (recomendado) y fp32, y `resultados/dano_v1/`. Nombres: detector `v1, v2, v3…`; zona dañada `dano_v1, dano_v2…` (el siguiente se llama `dano_v2`). Test con cajas de `v2`: IoU daño 0.64, error de gravedad ±9,6 puntos, 35/35 rajados con daño y 0/30 sanos. Documentado al final de `MODELO_IO.md`. Se eligió un segundo modelo y no YOLOv8-seg porque solo hay zona anotada en ~17 % de los Crack.
- **Pendiente:** validación con video real en la app, de los dos modelos. Si el detector falla, recoger esos frames para reentrenarlo; su siguiente run debe llamarse `v3` (`dano_v1` es otro modelo).
- La cuota gratis de GPU de Colab se agota tras ~3,5 h. Evaluar y exportar funciona en un runtime de CPU (`eggs_v2.ipynb` celdas 5 y 7–10).

## Preferencias de trabajo
- Explicar **en español** y **brevemente** qué hace cada celda.
- **No borrar ni sobrescribir** resultados anteriores: usar siempre **nombres de run nuevos** (nunca `exist_ok=True` sobre un run existente).
