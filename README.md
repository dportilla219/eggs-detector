# Detector de huevos rajados (YOLOv8 → TFLite)

Modelo de detección para clasificar huevos en **video en vivo** (frame a frame) dentro de una app React Native / Expo (Android e iOS) que usa `react-native-vision-camera` + `react-native-fast-tflite`.

| id | clase | significado |
|----|-------|-------------|
| 0 | `Crack` | huevo rajado |
| 1 | `Intact` | huevo sano |

## Entregable
- `.tflite` ligero para el celular.
- Documentación de entrada/salida del modelo (tamaño y formato de entrada, normalización, forma de la salida y cómo decodificarla).

*Pendiente: se agregarán cuando termine el entrenamiento y la exportación.*

## Contenido
- `eggs_train.ipynb`: notebook completo, pensado para Google Colab con GPU.
  1. Verificar GPU e instalar `ultralytics`.
  2. Montar Drive y descomprimir el dataset en `/content/eggs_v2`.
  3. Reescribir `data.yaml` con rutas absolutas.
  4. Conteo por split y clase + chequeo de integridad.
  5. Cuadrícula de ejemplos con sus cajas.
  6. Balanceo de clases (duplica Intact solo en train).
  7. Entrenamiento YOLOv8n (run `v1`).
  8. Reanudar entrenamiento desde `last.pt`.
  9. Métricas por clase y matriz de confusión en test.
  10. Curvas de pérdida y diagnóstico de overfitting.
  11. Cuadrícula de predicciones marcando errores.
  12. Umbral de confianza recomendado (curva F1).
  13. Diagnóstico y cambios sugeridos para el siguiente run.

## Datos y resultados
No se suben al repo. Viven en Google Drive:
- Dataset: `MyDrive/eggs_v2/eggs_v2_parte{1..4}_*.zip` (formato YOLOv8, sin duplicados entre splits).
- Runs de entrenamiento: `MyDrive/eggs_v2/runs/<nombre_del_run>`.

## Cómo ejecutarlo
1. Subir los 4 zips a `MyDrive/eggs_v2/` en Google Drive.
2. Abrir `eggs_train.ipynb` en Colab (o en VS Code con kernel de Colab) con GPU.
3. Ejecutar las celdas en orden. Cada run usa un nombre nuevo (`v1`, `v2`, ...) para no sobrescribir resultados anteriores.
