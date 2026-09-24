# Detector de huevos rajados (YOLOv8 → TFLite)

Modelo de detección para clasificar huevos en **video en vivo** (frame a frame) dentro de una app React Native / Expo (Android e iOS) que usa `react-native-vision-camera` + `react-native-fast-tflite`.

| id | clase | significado |
|----|-------|-------------|
| 0 | `Crack` | huevo rajado |
| 1 | `Intact` | huevo sano |

## Estado
**Modelo entregado: `v2`.** Exportado a TFLite, verificado contra el modelo original y probado localmente con la decodificación documentada. Falta validarlo con video real en la app.

## Entregable
| archivo | tamaño | uso |
|---|---|---|
| [`modelo/eggs_v2_fp32.tflite`](modelo/) | 12,3 MB | **Recomendado.** Delegado GPU: `'android-gpu'` en Android, `'core-ml'` en iOS. |
| [`modelo/eggs_v2_int8.tflite`](modelo/) | 3,3 MB | Alternativa si el delegado GPU falla o el modelo tiene que correr en CPU. Misma entrada y salida. |

- Entrada NHWC `[1, 640, 640, 3]` float32 RGB 0–1. Salida `[1, 6, 8400]`: `cx, cy, w, h` normalizados + score de Crack e Intact. Sin NMS. Umbral recomendado: **0.5**.
- **[MODELO_IO.md](MODELO_IO.md)**: entrada, salida, decodificación, código de ejemplo para el frame processor y recomendaciones para video en vivo.
- Copia de los modelos en Google Drive: `MyDrive/eggs_v2/exports/v2/`.

**¿Qué modelo usar?** Empezar con FP32 + GPU y medir los FPS en un celular de gama media. Con 15 o más inferencias por segundo alcanza. Si va lento o el delegado falla en algún dispositivo, cambiar a INT8: basta con cambiar el nombre del archivo.

## Resultados
| | v1 | v2 |
|---|---|---|
| Test original, mAP50-95 Crack | 0.974 | 0.970 |
| Test original, mAP50-95 Intact | 0.946 | 0.970 |
| Huevo sano fuera del montaje (sintético) | 0.68 | **0.99** |
| Huevo rajado dentro del montaje (sintético) | 0.98 | 1.00 |
| Huevos sobre fondos variados (sintético) | 0.72 | 0.98 |
| Huevo rajado, fotos reales del montaje | 0.85 | 0.85 |

Las filas "sintético" miden el acierto por imagen con conf 0.5. `.tflite` frente a `.pt` (test + sintéticas): mAP50 0.994 con FP32 y 0.987 con INT8, contra 0.993 del original. El detalle está en [`resultados/v2/resumen.json`](resultados/v2/resumen.json).

## Historial de modelos
| run | resumen |
|---|---|
| `v1` | YOLOv8n desde `yolov8n.pt`, 91 épocas (parada por `patience`). **Problema:** todas las fotos Intact son de un mismo montaje y todas las de otras fuentes son Crack, así que el modelo aprendió en parte a decidir por el fondo. Clasificaba como Crack 1 de cada 3 huevos sanos fuera del montaje. |
| `v2` ✅ | Fine-tune de `v1`, 60 épocas, con ~2.600 imágenes sintéticas (huevos intercambiados entre fondos) y sin las copias `_dup`. Quita el atajo del fondo sin empeorar el test original. |

## Contenido
- `modelo/`: los `.tflite` entregados.
- `MODELO_IO.md`: documentación para integrar el modelo en la app.
- `eggs_train.ipynb`: preparación del dataset, entrenamiento y evaluación de `v1`.
- `eggs_v2.ipynb`: se ejecuta de arriba abajo con *Run All*, sin depender del anterior. Las celdas que ya hicieron su trabajo se saltan solas.
  1. Setup (ultralytics, Drive, dataset). Funciona con o sin GPU.
  2. Generador de imágenes sintéticas: A = Intact sobre fondos de otras fuentes, B = Crack sobre el montaje, P = fondos procedurales.
  3. Dataset v2 (train/valid + sintéticas, `test_synth` aparte).
  4. Ejemplos de sintéticas.
  5. Diagnóstico del atajo del fondo en `v1`.
  6. Entrenamiento `v2` y celda para reanudar.
  7. Comparación `v1` vs `v2`, elección de modelo y umbral.
  8. Errores del modelo elegido.
  9. Export a LiteRT/TFLite con entrada NHWC (FP32 + INT8).
  10. Verificación `.pt` vs `.tflite` y `resumen.json`.
- `resultados/`: métricas por época de `v1` (`v1/results.csv`) y verificación del modelo entregado (`v2/resumen.json`).

## Datos y resultados en Drive
No se suben al repo:
- Dataset: `MyDrive/eggs_v2/eggs_v2_parte{1..4}_*.zip` (formato YOLOv8, sin duplicados entre splits).
- Runs de entrenamiento: `MyDrive/eggs_v2/runs/<run>` (`v1`, `v2` y sus evaluaciones).
- Modelos exportados: `MyDrive/eggs_v2/exports/<run>/`.

## Cómo ejecutarlo
1. Subir los 4 zips a `MyDrive/eggs_v2/` en Google Drive.
2. Abrir `eggs_v2.ipynb` en Colab (o en VS Code con kernel de Colab).
   - **Con GPU** para entrenar.
   - **Con CPU** basta para evaluar, exportar y verificar (celdas 5 y 7–10), por ejemplo cuando se agota la cuota de GPU de Colab.
3. *Run All*. Cada run usa un nombre nuevo (`v1`, `v2`, ...) y nada se sobrescribe.

## Próximos pasos
- Probar el modelo con video real en la app: otros fondos, iluminación, huevo en la mano, varios huevos juntos.
- Guardar los frames donde falle. Son los datos para un posible `v3`: fotos reales de huevos sanos fuera del montaje y grietas vistas desde varios ángulos.
