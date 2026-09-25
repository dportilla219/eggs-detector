# Modelos entregados

| archivo | tamaño | uso |
|---|---|---|
| `eggs_v4_fp32.tflite` | 12,3 MB | **Detector recomendado** (GPU en Android / Core ML en iOS) |
| `eggs_v4_int8.tflite` | 3,3 MB | Alternativa ligera para CPU |
| `eggs_v2_fp32.tflite` | 12,3 MB | Detector anterior (misma entrada y salida que `v4`) |
| `eggs_v2_int8.tflite` | 3,3 MB | Detector anterior, versión ligera |
| `eggs_dano_v1_fp16.tflite` | 4,9 MB | Extra: zona dañada y gravedad de cada huevo Crack (recomendado) |
| `eggs_dano_v1_fp32.tflite` | 9,6 MB | Lo mismo en FP32 |

Entrada, salida y ejemplo de integración: [../MODELO_IO.md](../MODELO_IO.md).
Copias originales en Google Drive: `MyDrive/eggs_v2/exports/v4/`, `MyDrive/eggs_v2/exports/v2/` y `MyDrive/eggs_v2/exports/v3_dano/` (nombre original de `dano_v1`; ahí los archivos se llaman `eggs_dano_v3_*`).
