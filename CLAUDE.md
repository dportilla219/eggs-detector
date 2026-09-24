# Proyecto: detector de huevos rajados (YOLOv8 → TFLite)

## Objetivo
- Entrenar un modelo **YOLOv8 de detección** de huevos que funcione en **video en vivo** (frame a frame, sin tomar fotos).
- Clases: `0 = Crack` (huevo rajado), `1 = Intact` (huevo sano).

## Alcance y entregable
- Mi parte es **solo el modelo**. Mis compañeros lo integran en una app **React Native / Expo** (Android e iOS) con `react-native-vision-camera` + `react-native-fast-tflite`.
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

## Preferencias de trabajo
- Explicar **en español** y **brevemente** qué hace cada celda.
- **No borrar ni sobrescribir** resultados anteriores: usar siempre **nombres de run nuevos** (nunca `exist_ok=True` sobre un run existente).
