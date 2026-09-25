# Detector de huevos rajados (YOLOv8 → TFLite)

Modelo de detección para clasificar huevos en **video en vivo** (frame a frame) dentro de una app React Native / Expo (Android e iOS) que usa `react-native-vision-camera` + `react-native-fast-tflite`.

| id | clase | significado |
|----|-------|-------------|
| 0 | `Crack` | huevo rajado |
| 1 | `Intact` | huevo sano |

## Estado
**Modelo entregado: `v4`** (reemplaza a `v2`, con la misma entrada y salida: solo cambia el nombre del archivo). `v2` acertaba el test original, pero en fotos reales de otras fuentes marcaba como rajados casi todos los huevos sanos. `v4` se reentrenó añadiendo fotos reales y ahora acierta en fotos que no vio al entrenar: 36/39 huevos rajados y 18/20 sanos del dataset público, 97/105 huevos sanos de Wikimedia Commons, sin empeorar el test original (acierto por imagen 0.982 frente a 0.976 de `v2`). Detalle en [Resultados](#resultados).

**Extra: `dano_v1`.** Es un segundo modelo, opcional, que recibe el recorte de cada huevo rajado y devuelve la **zona dañada**. Con ella la app pinta dónde está el daño y calcula la **gravedad** (leve / media / grave). En test, con las cajas de `v2`, el IoU de la zona es 0.64 y el error de la gravedad ±9,6 puntos; marca daño en 35/35 rajados y en 0/30 sanos.

**App web desplegada: https://34-225-169-137.sslip.io** (la dirección cambia si la instancia de AWS cambia de IP; ver [`app/README.md`](app/README.md)). Los dos modelos corren en un servidor (AWS EC2) con una **simulación de banda transportadora**: cada huevo pasa por la cámara, `v2` lo clasifica, `dano_v1` mide el daño de los rajados y un desviador lo manda a empaque, industria o descarte según la gravedad. Se abre desde cualquier navegador, también en el celular. Código y detalles en [`app/`](app/README.md).

Hay también una **versión sin servidor** de la misma app, que ejecuta los modelos en el navegador (`app/web/local.js`), para cuando la instancia está apagada.

## Entregable
| archivo | tamaño | uso |
|---|---|---|
| [`modelo/eggs_v4_fp32.tflite`](modelo/) | 12,3 MB | **Recomendado.** Delegado GPU: `'android-gpu'` en Android, `'core-ml'` en iOS. |
| [`modelo/eggs_v4_int8.tflite`](modelo/) | 3,3 MB | Alternativa si el delegado GPU falla o el modelo tiene que correr en CPU. Misma entrada y salida. |
| [`modelo/eggs_v2_fp32.tflite`](modelo/), [`eggs_v2_int8.tflite`](modelo/) | 12,3 / 3,3 MB | Versión anterior del detector, misma entrada y salida. Solo acierta con huevos sanos del montaje del dataset. |
| [`modelo/eggs_dano_v1_fp16.tflite`](modelo/) | 4,9 MB | Extra opcional: zona dañada y gravedad de cada huevo Crack. Entrada `[1,192,192,3]` (recorte del huevo); salida `[1,192,192,2]` (huevo, daño). |
| [`modelo/eggs_dano_v1_fp32.tflite`](modelo/) | 9,6 MB | Lo mismo en FP32 (referencia). |

- Entrada NHWC `[1, 640, 640, 3]` float32 RGB 0–1. Salida `[1, 6, 8400]`: `cx, cy, w, h` normalizados + score de Crack e Intact. Sin NMS. Umbral recomendado: **0.5**.
- ⚠️ **Se usa un Expo development build (`expo-dev-client`), no Expo Go**: la cámara y TFLite son módulos nativos que Expo Go no incluye. Los pasos están en MODELO_IO.md.
- **[MODELO_IO.md](MODELO_IO.md)**: entrada, salida, decodificación, código de ejemplo para el frame processor y recomendaciones para video en vivo. Al final está la sección del modelo de zona dañada: recorte, máscaras, gravedad y su propio ejemplo.
- Copia de los modelos en Google Drive: `MyDrive/eggs_v2/exports/v4/` (y `exports/v2/`).

**Dos modelos:** el detector (`v4`) es obligatorio. `dano_v1` es un extra opcional que va detrás del detector y no lo reemplaza (ver *Qué modelos usar* en la guía). La guía y `MODELO_IO.md` hablan de `v2` porque se escribieron con él: todo vale igual para `v4` cambiando el nombre del archivo.

**¿FP32 o INT8?** Empezar con FP32 + GPU y medir los FPS en un celular de gama media. Con 15 o más inferencias por segundo alcanza. Si va lento o el delegado falla en algún dispositivo, cambiar a INT8: basta con cambiar el nombre del archivo.

---

## Guía para el equipo de la app

### Qué modelos usar: `v2` siempre, `dano_v1` si quieren el extra

Son **dos modelos distintos que trabajan juntos**. `dano_v1` **no reemplaza** a `v2`.

| modelo | archivo | ¿obligatorio? | qué hace | cuándo se ejecuta |
|---|---|---|---|---|
| **`v2`** (detector; usar `v4`, que es igual por fuera) | `eggs_v4_fp32.tflite` | **Sí** | Encuentra cada huevo y dice si está rajado (`Crack`) o sano (`Intact`). | En cada frame, ~10–15 por segundo. |
| **`dano_v1`** (zona dañada) | `eggs_dano_v1_fp16.tflite` | No, es el extra | Marca dónde está el daño y calcula la gravedad (leve / media / grave). | Solo en los huevos que `v2` dio como `Crack`, ~3–5 veces por segundo por huevo. |

**Orden de trabajo recomendado:**
1. Integrar **solo `v2`** y comprobar que funciona (pasos 1–9 de abajo).
2. Después, si quieren el extra, añadir **`dano_v1`** (paso 10). No hay que tocar nada de lo que ya hace `v2`.

El nombre de los modelos no indica cuál es más nuevo. `v1, v2, v3…` son versiones del **detector**; `dano_v1, dano_v2…` son versiones del modelo de **zona dañada**. Si un día llega un `v3`, sustituirá a `v2`, y `dano_v1` seguirá funcionando detrás de él.

### Por qué no se puede usar Expo Go

Expo Go es una app ya compilada que solo trae los módulos nativos que Expo decidió incluir. Para analizar video en vivo con un modelo hacen falta tres librerías **nativas** que Expo Go no trae:

| librería | para qué |
|---|---|
| `react-native-vision-camera` | acceder a cada frame de la cámara en tiempo real (`expo-camera` no da los frames) |
| `vision-camera-resize-plugin` | recortar y escalar el frame a 640×640 RGB |
| `react-native-fast-tflite` | ejecutar el `.tflite` en el celular (GPU / Core ML) |

Si abren el proyecto en Expo Go, la app falla al cargar la cámara o el modelo. La solución es un **development build**: su propia versión de "Expo Go" con estas librerías incluidas. Se instala una vez en el celular y después se trabaja **igual que con Expo Go**: `npx expo start`, escanear el QR y los cambios se recargan solos. Solo hay que volver a generar la app si se agrega otra librería nativa.

El modelo corre **dentro del celular**. No hace falta backend para la detección; el backend solo sirve si quieren guardar historial o resultados.

### Requisitos
- Node LTS y un proyecto Expo (SDK reciente).
- **Android:** un celular Android real con *depuración USB*, y Android Studio (para `expo run:android`) **o** una cuenta de Expo (para compilar en la nube con EAS).
- **iOS:** un iPhone real (el simulador no tiene cámara) y una Mac con Xcode **o** EAS Build. Instalar en un iPhone físico con EAS requiere una cuenta de Apple Developer.

### Pasos

**1. Copiar el modelo al proyecto de la app**

Copiar [`modelo/eggs_v2_fp32.tflite`](modelo/) (y opcionalmente `eggs_v2_int8.tflite`) a `assets/models/` del proyecto de la app. Si van a usar el extra, copien también `eggs_dano_v1_fp16.tflite`.

**2. Instalar las librerías**
```bash
npx expo install expo-dev-client react-native-vision-camera react-native-worklets-core vision-camera-resize-plugin react-native-fast-tflite
```

**3. Configurar Babel** (`babel.config.js`). Los frame processors se ejecutan como *worklets*:
```js
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [['react-native-worklets-core/plugin']],
  };
};
```

**4. Permitir archivos `.tflite` en Metro** (`metro.config.js`; si no existe, crearlo con `npx expo customize metro.config.js`):
```js
const { getDefaultConfig } = require('expo/metro-config');
const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('tflite');
module.exports = config;
```

**5. Activar los plugins en `app.json`**:
```json
{
  "expo": {
    "plugins": [
      ["react-native-vision-camera", { "cameraPermissionText": "Se usa la cámara para revisar los huevos." }],
      ["react-native-fast-tflite", { "enableCoreMLDelegate": true, "enableAndroidGpuLibraries": true }]
    ]
  }
}
```
Las opciones exactas de cada plugin pueden cambiar entre versiones; revísenlas en el README de cada librería.

**6. Generar e instalar el development build** (una sola vez y cada vez que agreguen otra librería nativa):
```bash
# Opción local (celular conectado por USB)
npx expo prebuild
npx expo run:android            # o: npx expo run:ios --device

# Opción en la nube, sin Android Studio / Xcode
npm i -g eas-cli && eas login
eas build --profile development --platform android   # instala el APK que genera
```

**7. Trabajar día a día**
```bash
npx expo start --dev-client
```
Abrir la app instalada en el paso 6 (no Expo Go) y escanear el QR.

**8. Implementar la detección**

El código completo del frame processor (recorte, redimensionado, inferencia, decodificación y NMS) está en [MODELO_IO.md → Ejemplo](MODELO_IO.md#ejemplo-frame-processor). Además:
- Pedir el permiso con `useCameraPermission()` y mostrar `<Camera device={...} isActive frameProcessor={...} />`.
- Limitar la inferencia a ~10–15 por segundo con `runAtTargetFps`.
- Para actualizar la interfaz desde el frame processor, pasar los resultados al hilo de JS con `Worklets.createRunOnJS(...)` de `react-native-worklets-core`.
- Suavizar el resultado: decidir "rajado / sano" por mayoría en los últimos 5–10 frames, no con un solo frame.

**9. Comprobar que funciona**
- [ ] La app abre la cámara en el development build (no en Expo Go).
- [ ] Con un huevo centrado, el mejor candidato tiene `cx ≈ cy ≈ 0.5` y confianza > 0.5.
- [ ] Un huevo sano sale `Intact` y uno rajado sale `Crack`.
- [ ] Medir los FPS reales. Si bajan de ~10, probar el modelo INT8 o bajar `runAtTargetFps`.

**10. (Opcional) Añadir la zona dañada con `dano_v1`**

Solo cuando `v2` ya funcione. El código está en [MODELO_IO.md → Modelo 2](MODELO_IO.md#modelo-2-zona-dañada-y-gravedad-dano_v1). En resumen:
1. Cargar el segundo modelo con otro `useTensorflowModel(require('../assets/models/eggs_dano_v1_fp16.tflite'), ...)`. No hacen falta librerías nuevas ni regenerar el development build.
2. En el mismo frame processor, después de la NMS de `v2`, para cada huevo `Crack`:
   - pasar su caja a píxeles del **frame completo** y expandirla un 5 % por lado;
   - recortar y **estirar** a 192×192 RGB 0–1 con `resize`;
   - ejecutar `dano_v1`.
3. Contar los píxeles de huevo (canal 0 > 0.5) y de daño (canal 1 > 0.5 dentro del huevo). La gravedad es `daño / huevo`: leve < 15 %, media < 35 %, grave ≥ 35 %.
4. Dibujar solo el canal de daño sobre el huevo y mostrar el nivel. Suavizar la gravedad con la media de los últimos 5–10 valores.

Comprobar:
- [ ] Un huevo sano no muestra zona dañada: `dano_v1` ni siquiera se ejecuta, porque `v2` lo da como `Intact`.
- [ ] En un huevo rajado, la mancha cae sobre la grieta y no se desplaza. Si aparece corrida, revisar que el recorte se hace sobre el frame completo con la caja +5 % por lado.
- [ ] Los FPS no bajan de forma notable. Si bajan, ejecutar `dano_v1` con menos frecuencia (2–3 por segundo) o solo en el huevo más grande.

### Problemas frecuentes
| síntoma | causa probable |
|---|---|
| La app se cierra o dice que falta un módulo nativo | Se abrió en Expo Go, o no se regeneró el build tras instalar librerías. |
| `Unable to resolve module ...tflite` | Falta `assetExts.push('tflite')` en `metro.config.js` (hay que reiniciar Metro). |
| Cajas desplazadas o todo sale de una clase | El frame no llega como RGB 0–1 (se pasó BGR o 0–255) o no se recortó un cuadrado. |
| Error del delegado GPU en algún Android | Usar el delegado por defecto (CPU) con `eggs_v2_int8.tflite`. |
| Va lento | INT8, `runAtTargetFps(10, ...)`, y no dibujar en cada frame. |
| La zona dañada aparece desplazada o deformada | El recorte de `dano_v1` no se hizo sobre el frame completo, falta el 5 % por lado o se mantuvo la proporción en lugar de estirar a 192×192. |
| `dano_v1` marca manchas fuera del huevo | Se está dibujando el canal 0 (silueta) o el daño fuera de la silueta. Dibujar solo el canal 1 donde el canal 0 > 0.5. |

### Prompt de contexto para su asistente de IA

Copien esto al inicio de la conversación con la IA que usen (Claude, ChatGPT, Copilot…) para que tenga el contexto correcto:

```text
Estoy integrando un modelo de visión por computadora en una app React Native con Expo (Android e iOS).

OBJETIVO
- Detectar huevos en VIDEO EN VIVO desde la cámara (frame a frame, sin tomar fotos) y
  clasificarlos como 0 = Crack (rajado) o 1 = Intact (sano).
- La inferencia corre EN EL CELULAR. No hay backend para la detección.

RESTRICCIONES (no las cambies)
- NO se usa Expo Go: usamos un Expo development build (expo-dev-client), porque las
  librerías son nativas. No me propongas soluciones que requieran Expo Go ni expo-camera
  para la detección.
- Librerías: react-native-vision-camera (frame processors), react-native-worklets-core,
  vision-camera-resize-plugin y react-native-fast-tflite.
- El modelo ya está entrenado y exportado; no hay que reentrenarlo ni convertirlo.

MODELO (YOLOv8n exportado a TFLite)
- Archivo: assets/models/eggs_v2_fp32.tflite (recomendado, delegado 'android-gpu' en
  Android y 'core-ml' en iOS). Alternativa: eggs_v2_int8.tflite (CPU), misma entrada y salida.
- Entrada: 1 tensor float32 de forma [1, 640, 640, 3], NHWC, RGB, valores 0..1.
  Se obtiene recortando un CUADRADO centrado del frame y escalándolo a 640x640 con
  vision-camera-resize-plugin: { crop, scale: {width: 640, height: 640}, pixelFormat: 'rgb', dataType: 'float32' }.
- Salida: 1 tensor float32 [1, 6, 8400] (Float32Array plano de 6*8400).
  El valor c del candidato i está en out[c * 8400 + i]:
    c=0 cx, c=1 cy, c=2 w, c=3 h  -> caja normalizada 0..1 respecto al cuadrado de 640
    c=4 score Crack, c=5 score Intact -> ya son probabilidades 0..1 (sigmoide aplicada)
  NO incluye NMS.
- Decodificación: para cada i, clase = argmax(score Crack, score Intact), conf = ese score;
  descartar conf < 0.5; NMS agnóstico a la clase con IoU 0.5; convertir la caja a píxeles
  del cuadrado recortado (x = crop.x + cx * lado, etc.).
- Recomendaciones: runAtTargetFps ~10-15; pasar resultados al hilo JS con
  Worklets.createRunOnJS; decidir la clase por mayoría en los últimos 5-10 frames.

MODELO 2 OPCIONAL: ZONA DAÑADA (dano_v1, eggs_dano_v1_fp16.tflite)
- Son DOS modelos que trabajan juntos: v2 es obligatorio y detecta/clasifica; dano_v1 es un
  extra que va DESPUÉS de v2 y NO lo reemplaza. Primero haz funcionar v2 solo; luego añade dano_v1.
- Solo para huevos clasificados Crack. Entrada float32 [1, 192, 192, 3] NHWC RGB 0..1:
  la caja del huevo en píxeles del FRAME COMPLETO, expandida 5 % por lado (ancho y alto),
  recortada al frame y ESTIRADA a 192x192 (resize-plugin: crop + scale, sin mantener proporción).
- Salida float32 [1, 192, 192, 2]; píxel (x,y) canal c en out[(y*192+x)*2+c].
  c=0 silueta del huevo, c=1 zona dañada; > 0.5 = sí.
- Gravedad = píxeles (daño y huevo) / píxeles huevo. Leve < 0.15, media < 0.35, grave >= 0.35.
- Dibujar solo el canal de daño (no el contorno del huevo) sobre el rectángulo recortado.
- Ejecutarlo 3-5 veces por segundo por huevo y suavizar la gravedad en 5-10 valores.

REFERENCIA
- La documentación completa y un ejemplo de frame processor están en MODELO_IO.md del
  repositorio del modelo: https://github.com/dportilla219/eggs-detector

Antes de escribir código, verifica en la documentación oficial las APIs exactas de las
versiones que tengo instaladas (vision-camera v4+, fast-tflite, resize-plugin), porque
cambian entre versiones.

Lo que necesito ahora es: <describir la tarea>
```

## Resultados
| | v1 | v2 | v3 | **v4** |
|---|---|---|---|---|
| Test original, mAP50-95 Crack | 0.974 | 0.970 | 0.969 | 0.964 |
| Test original, mAP50-95 Intact | 0.946 | 0.970 | 0.974 | 0.961 |
| Huevo sano fuera del montaje (sintético) | 0.68 | 0.99 | 1.00 | 1.00 |
| Huevo rajado dentro del montaje (sintético) | 0.98 | 1.00 | 1.00 | 1.00 |
| Huevos sobre fondos variados (sintético) | 0.72 | 0.98 | 0.98 | 0.94 |
| Huevo rajado, fotos reales del montaje | 0.85 | 0.85 | 0.85 | 0.85 |
| Huevo sano, fotos reales del montaje | | 0.98 | 0.98 | **1.00** |
| **Fotos reales de otra fuente, rajados** (39) | | 0.80 | 0.62 | **0.92** |
| **Fotos reales de otra fuente, sanos** (20) | | 0.00 | 1.00 | **0.90** |
| **Wikimedia Commons, sanos** (105) | | 0.00 | 0.98 | **0.92** |
| **Wikimedia Commons, rajados** (7) | | 0.43 | 0.00 | 0.43 |

Acierto por imagen con conf 0.5 salvo las filas de mAP. Las cuatro últimas filas son fotos que ningún modelo vio al entrenar: el test del dataset público [Egg-Defect-Detection](https://github.com/dakshkathuria346-gif/Egg-Defect-Detection) (fotos de celular) y un test independiente de 112 fotos de Wikimedia Commons elegidas a mano, de autores distintos a los de entrenamiento. Las fotos reales se midieron con la tubería de la app (`.tflite` en CPU): [`resultados/v4/verificacion_tflite_app.json`](resultados/v4/verificacion_tflite_app.json). `.tflite` FP32 de `v4` frente a su `.pt` (test + sintéticas): mAP50 0.991 contra 0.992. Detalle en `resultados/v2/`, `resultados/v3/` y `resultados/v4/`.

Límite conocido: solo hay 7 fotos de huevos rajados en Commons, así que esa fila tiene mucha incertidumbre; las grietas finas o de espaldas a la cámara se pueden pasar por sanas.

## Historial de modelos
| run | resumen |
|---|---|
| `v1` | YOLOv8n desde `yolov8n.pt`, 91 épocas (parada por `patience`). **Problema:** todas las fotos Intact son de un mismo montaje y todas las de otras fuentes son Crack, así que el modelo aprendió en parte a decidir por el fondo. Clasificaba como Crack 1 de cada 3 huevos sanos fuera del montaje. |
| `v2` ✅ | Fine-tune de `v1`, 60 épocas, con ~2.600 imágenes sintéticas (huevos intercambiados entre fondos) y sin las copias `_dup`. Quita el atajo del fondo sin empeorar el test original. |
| `v3` | Fine-tune de `v2` con fotos reales de otras fuentes (dataset público con cajas de Grounding DINO y fotos de Commons de huevos sanos). Aprendió los huevos sanos reales, pero empeoró con los rajados de otra fuente: el split por bloques dejó un grupo entero de fotos (`Egg_X`) solo en test. No se entregó. |
| `v4` ✅ | Fine-tune de `v3` con el split intercalado por subgrupo (cada tipo de foto aparece en train, valid y test). Se elige por la **peor clase** en fotos reales, no por el promedio. Se exportó `last.pt` (época 9, parada por `patience`). Es el detector entregado. |
| `dano_v1` ✅ | Segundo modelo (MobileNetV2-0.5 + U-Net, 80 épocas) que segmenta la silueta y la zona dañada en el recorte de cada huevo. Anotaciones propias: 420 huevos rajados marcados sobre una rejilla 10×10 y siluetas con SAM 2.1. No sustituye a `v2`: va detrás de él. |

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
- `eggs_dano.ipynb`: modelo de zona dañada, ejecutado de arriba abajo en Colab con GPU. Genera el dataset de recortes con SAM y las anotaciones, lo revisa visualmente, entrena `dano_v1`, exporta a TFLite (FP32 + FP16) y evalúa la tubería completa `v2` → recorte → `dano_v1`.
- `dano/`: scripts que usa ese notebook y las anotaciones (`dano/anotaciones/`). Cada línea de `ann_*.txt` indica las celdas dañadas de un huevo, por ejemplo `12: B3-5 C4`.
- `resultados/`: métricas por época de `v1` (`v1/results.csv`), verificación del modelo entregado (`v2/resumen.json`) y evaluación del modelo de zona dañada (`dano_v1/resumen.json`, `dano_v1/ejemplos_test.jpg`).
- `eggs_v3.ipynb`: reentrenamiento del detector con fotos reales de huevos fuera del montaje (Colab con GPU, *Ejecutar todo*). Compara `v2` y `v3` en el test original, en fotos reales públicas y en un test independiente de Wikimedia Commons.
- `eggs_v4.ipynb`: `v4` a partir de `v3`, reutilizando las cajas de las fotos reales con otro split. Compara `v2`, `v3` y `v4` y exporta el elegido.
- `app/`: app web desplegada (FastAPI + LiteRT) con la simulación de la banda transportadora. Ver [`app/README.md`](app/README.md).

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
- Guardar los frames donde falle. Son los datos para una siguiente versión del detector: fotos reales de huevos sanos fuera del montaje y grietas vistas desde varios ángulos.
- Zona dañada: comprobar en video que la gravedad no salta entre frames y ajustar los umbrales leve / media / grave con huevos reales.
