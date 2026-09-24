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

## Preferencias de trabajo
- Explicar **en español** y **brevemente** qué hace cada celda.
- **No borrar ni sobrescribir** resultados anteriores: usar siempre **nombres de run nuevos** (nunca `exist_ok=True` sobre un run existente).
