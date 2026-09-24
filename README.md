# Detector de huevos rajados (YOLOv8 → TFLite)

Modelo de detección para clasificar huevos en **video en vivo** (frame a frame) dentro de una app React Native / Expo (Android e iOS) que usa `react-native-vision-camera` + `react-native-fast-tflite`.

| id | clase | significado |
|----|-------|-------------|
| 0 | `Crack` | huevo rajado |
| 1 | `Intact` | huevo sano |

## Entregable
- [`modelo/eggs_v2_fp32.tflite`](modelo/) (12,3 MB, recomendado) y `eggs_v2_int8.tflite` (3,3 MB), con entrada NHWC `[1, 640, 640, 3]` y salida `[1, 6, 8400]`. También están en `MyDrive/eggs_v2/exports/v2/`.
- **[MODELO_IO.md](MODELO_IO.md)**: entrada, salida, decodificación y ejemplo de integración para la app.

## Historial de modelos
| run | resumen |
|---|---|
| `v1` | YOLOv8n, 91 épocas. Test: mAP50 0.982 / mAP50-95 0.960. **Problema:** todas las fotos Intact son de un mismo montaje y todas las de otras fuentes son Crack, así que el modelo puede decidir por el fondo. |
| `v2` ✅ | Fine-tune de `v1` con imágenes sintéticas (huevos intercambiados entre fondos) y sin copias `_dup`. Test original: mAP50-95 0.970 en ambas clases. Huevos sanos fuera del montaje: acierto 0.68 → **0.99**. **Es el modelo entregado.** |

## Contenido
- `eggs_train.ipynb`: preparación del dataset, entrenamiento y evaluación de `v1`.
- `eggs_v2.ipynb`: se ejecuta de arriba abajo, sin depender del anterior.
  1. Setup (GPU, ultralytics, Drive, dataset).
  2. Generador de imágenes sintéticas: A = Intact sobre fondos de otras fuentes, B = Crack sobre el montaje, P = fondos procedurales.
  3. Dataset v2 (train/valid + sintéticas, `test_synth` aparte).
  4. Ejemplos de sintéticas.
  5. Diagnóstico del atajo del fondo en `v1`.
  6. Entrenamiento `v2` y celda para reanudar.
  7. Comparación `v1` vs `v2`, elección de modelo y umbral.
  8. Errores del modelo elegido.
  9. Export a LiteRT/TFLite con entrada NHWC (FP32 + INT8).
  10. Verificación `.pt` vs `.tflite` y `resumen.json` para la app.
- `resultados/`: métricas por época (`v1/results.csv`) y verificación del modelo entregado (`v2/resumen.json`).

## Datos y resultados
No se suben al repo. Viven en Google Drive:
- Dataset: `MyDrive/eggs_v2/eggs_v2_parte{1..4}_*.zip` (formato YOLOv8, sin duplicados entre splits).
- Runs de entrenamiento: `MyDrive/eggs_v2/runs/<run>`.
- Modelos exportados: `MyDrive/eggs_v2/exports/<run>/`.

## Cómo ejecutarlo
1. Subir los 4 zips a `MyDrive/eggs_v2/` en Google Drive.
2. Abrir el notebook en Colab (o en VS Code con kernel de Colab) con GPU.
3. Ejecutar las celdas en orden. Cada run usa un nombre nuevo (`v1`, `v2`, ...) para no sobrescribir resultados anteriores.
