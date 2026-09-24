# Modelo de huevos: entrada, salida e integración en la app

Guía para integrar el detector en la app React Native / Expo con `react-native-vision-camera` + `react-native-fast-tflite`.

Modelo: **`v2`** (YOLOv8n). Los archivos están en Google Drive, en `MyDrive/eggs_v2/exports/v2/`, y en la carpeta [`modelo/`](modelo/) de este repo. Los datos de la verificación están en [`resultados/v2/resumen.json`](resultados/v2/resumen.json).

## ⚠️ Development build, no Expo Go

Decisión del equipo: la app se ejecuta en un **Expo development build**.

`react-native-vision-camera`, `react-native-fast-tflite` y `vision-camera-resize-plugin` son **módulos nativos**. Expo Go no los incluye, así que la app fallará al abrir la cámara o cargar el modelo. Esto no depende del modelo: pasa con cualquier `.tflite`.

Hay que usar un **development build**, que se instala en el celular y se usa casi igual que Expo Go, con recarga en caliente:

```bash
npx expo install expo-dev-client react-native-vision-camera react-native-worklets-core vision-camera-resize-plugin react-native-fast-tflite
npx expo prebuild
npx expo run:android      # o: npx expo run:ios
# sin Android Studio / Xcode: eas build --profile development
```

En `app.json`, activar los plugins. Revisen las opciones exactas en el README de cada librería para la versión que instalen:

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

El modelo corre **en el celular**. No hace falta un backend para la inferencia; el backend solo se necesita si quieren guardar resultados o historial.

## Archivos

| archivo | tamaño | cuándo usarlo |
|---|---|---|
| `eggs_v2_fp32.tflite` | 12,3 MB | **Recomendado.** Con delegado GPU (Android) o Core ML (iOS), que lo ejecutan en FP16. Da los mismos resultados que el modelo original. |
| `eggs_v2_int8.tflite` | 3,3 MB | Alternativa ligera para CPU. Clasifica casi igual, pero sus cajas son algo menos precisas (ver *Rendimiento*). |

Ambos tienen **la misma entrada y salida**. Se pueden cambiar sin tocar el código de decodificación.

## Clases

| id | nombre | significado |
|---|---|---|
| 0 | `Crack` | huevo rajado |
| 1 | `Intact` | huevo sano |

## Entrada

| | |
|---|---|
| forma | `[1, 640, 640, 3]` (NHWC: alto, ancho, canal) |
| tipo | `float32` |
| color | **RGB** (no BGR) |
| rango | `0.0 – 1.0` (píxel / 255) |

El modelo espera una imagen cuadrada. Un frame de cámara no es cuadrado (por ejemplo 1920×1080). Si se estira directamente a 640×640, los huevos se deforman y el modelo rinde peor. Hay que **recortar un cuadrado centrado** y escalarlo a 640×640, y las cajas quedan en coordenadas de ese cuadrado.

## Salida

| | |
|---|---|
| forma | `[1, 6, 8400]` |
| tipo | `float32` |
| NMS | **no incluido**, hay que hacerlo en la app (ver abajo) |

Son 8400 candidatos (80×80 + 40×40 + 20×20 celdas). En el `Float32Array` plano, el valor `c` del candidato `i` está en `out[c * 8400 + i]`:

| fila `c` | contenido |
|---|---|
| 0 | `cx`: centro x de la caja, **normalizado 0–1** respecto a la entrada de 640 |
| 1 | `cy`: centro y, normalizado 0–1 |
| 2 | `w`: ancho, normalizado 0–1 |
| 3 | `h`: alto, normalizado 0–1 |
| 4 | score de `Crack`, ya en 0–1 (sigmoide aplicada) |
| 5 | score de `Intact`, ya en 0–1 |

## Decodificación

1. Para cada candidato `i`, la clase es la de mayor score (fila 4 o 5) y la confianza es ese score.
2. Se descartan los candidatos con confianza `< CONF` (**0.5**; la curva F1 da 0.52 y el rendimiento es casi igual en un rango amplio, 0.1–0.9).
3. NMS **agnóstico a la clase** con IoU 0.5: ordenar por confianza y descartar las cajas que solapen más de 0.5 con una ya aceptada, sin importar su clase. Así un mismo huevo no sale a la vez como Crack e Intact.
4. Convertir a píxeles del cuadrado recortado: `x1 = (cx - w/2) * lado`, `y1 = (cy - h/2) * lado`, etc. Después sumar el offset del recorte para llevarlas al frame.

## Ejemplo (frame processor)

```ts
import { useFrameProcessor } from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useTensorflowModel } from 'react-native-fast-tflite';

const N = 8400;
const CONF = 0.5;
const IOU = 0.5;

function iou(a: number[], b: number[]) {
  'worklet';
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

export function useEggDetector() {
  const tflite = useTensorflowModel(require('../assets/eggs_v2_fp32.tflite'), 'android-gpu'); // iOS: 'core-ml'
  const model = tflite.state === 'loaded' ? tflite.model : undefined;
  const { resize } = useResizePlugin();

  return useFrameProcessor((frame) => {
    'worklet';
    if (model == null) return;
    const side = Math.min(frame.width, frame.height);
    const crop = { x: (frame.width - side) / 2, y: (frame.height - side) / 2, width: side, height: side };
    const input = resize(frame, {
      crop,
      scale: { width: 640, height: 640 },
      pixelFormat: 'rgb',
      dataType: 'float32', // valores 0..1
    });
    const out = model.runSync([input])[0] as Float32Array;

    const cands: { box: number[]; conf: number; cls: number }[] = [];
    for (let i = 0; i < N; i++) {
      const sCrack = out[4 * N + i];
      const sIntact = out[5 * N + i];
      const conf = Math.max(sCrack, sIntact);
      if (conf < CONF) continue;
      const cx = out[i], cy = out[N + i], w = out[2 * N + i], h = out[3 * N + i];
      cands.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], conf, cls: sCrack >= sIntact ? 0 : 1 });
    }
    cands.sort((a, b) => b.conf - a.conf);
    const eggs: typeof cands = [];
    for (const c of cands) {
      if (eggs.every((e) => iou(e.box, c.box) < IOU)) eggs.push(c);
    }
    // eggs[k].box está normalizado 0..1 respecto al cuadrado `crop`:
    // px = crop.x + box[0] * side, py = crop.y + box[1] * side, ...
    // eggs[k].cls: 0 = Crack, 1 = Intact
  }, [model]);
}
```

Configuración de Metro para empaquetar el modelo (`metro.config.js`):

```js
config.resolver.assetExts.push('tflite');
```

## Recomendaciones para video en vivo

- **Suavizado temporal:** no mostrar la clase de un solo frame. Por ejemplo, decidir por mayoría en los últimos 5–10 frames, o con una media móvil del score de Crack. Así se evita que la etiqueta parpadee.
- **Frecuencia:** no hace falta inferir en todos los frames. Con 10–15 inferencias por segundo es suficiente y ahorra batería (`runAtTargetFps` de vision-camera).
- **Delegado:** empezar con `fp32` + `'android-gpu'` / `'core-ml'`. Si en algún dispositivo falla, pasar a `int8` con el delegado por defecto (CPU).
- **Comprobación rápida:** con la cámara apuntando a un huevo centrado, el candidato de mayor score debería tener `cx ≈ cy ≈ 0.5`. Si las cajas salen desplazadas o las clases al revés, revisar que el formato sea RGB y no BGR, y que el rango sea 0–1 y no 0–255.

## Rendimiento medido

**Test original** (382 fotos reales que el modelo no vio al entrenar):

| clase | precisión | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Crack | 0.990 | 0.958 | 0.981 | 0.970 |
| Intact | 0.932 | 0.976 | 0.988 | 0.970 |

**Acierto por imagen** con conf 0.5. Las imágenes sintéticas son huevos del test recortados y pegados sobre otros fondos, para comprobar que el modelo mira el huevo y no el fondo:

| grupo | acierto |
|---|---|
| Huevo sano fuera del montaje de entrenamiento (sintético) | 0.99 |
| Huevo rajado dentro del montaje (sintético) | 1.00 |
| Huevos sobre fondos variados (sintético) | 0.98 |
| Huevo rajado, fotos reales de otras fuentes | 0.98 |
| Huevo sano, fotos reales del montaje | 0.98 |
| Huevo rajado, fotos reales del montaje | 0.85 |

El grupo más débil son las fotos del montaje donde la grieta casi no se ve (del otro lado o muy fina a 224 px). En video, al girar el huevo, conviene que la grieta quede a la vista de la cámara.

**`.tflite` frente al modelo original** (test original + sintéticas, 918 imágenes):

| archivo | mAP50 | mAP50-95 |
|---|---|---|
| modelo original `.pt` | 0.993 | 0.986 |
| `eggs_v2_fp32.tflite` | 0.994 | 0.969 |
| `eggs_v2_int8.tflite` | 0.987 | 0.837 |

El mAP50 mide sobre todo si acierta la clase, y es prácticamente igual en los tres. El mAP50-95 mide lo ajustada que queda la caja: FP32 está muy cerca y INT8 es más aproximado. Para decir "rajado / sano" los dos sirven.

**Velocidad:** en la CPU de Colab (x86) ambos tardan ~200 ms por imagen, pero no es representativo. En el teléfono, con el delegado GPU o Core ML, se espera bastante menos, e INT8 es más rápido que FP32 en CPU ARM. Hay que medirlo en el dispositivo real.

## Limitaciones conocidas

- Todas las fotos reales de huevos sanos vienen de un único montaje (fondo gris, base negra). Para compensarlo se entrenó con imágenes sintéticas (huevos recortados sobre otros fondos), pero **el modelo no se ha validado aún con video real** en otros fondos e iluminaciones. Conviene probarlo pronto y, si falla, guardar esos frames para reentrenar.
- Entrenado sobre todo con un huevo por imagen. Con varios huevos juntos (por ejemplo en un cartón) puede rendir peor.

---

# Modelo 2: zona dañada y gravedad (`v3_dano`)

Es un segundo modelo, **opcional**, que se usa **junto con `v2`**. `v2` encuentra cada huevo y dice si está rajado. `v3_dano` recibe el **recorte de un huevo** y devuelve dos máscaras: la silueta del huevo y la **zona dañada**. Con ellas la app puede:

- **pintar dónde está el daño** encima del huevo, en el video;
- calcular la **gravedad**: el % de la cáscara que está dañada (leve / media / grave).

`v2` no cambia: su entrada, su salida y su código siguen igual.

## Archivos

| archivo | tamaño | cuándo usarlo |
|---|---|---|
| `eggs_dano_v3_fp16.tflite` | 4,9 MB | **Recomendado.** Da los mismos resultados que FP32 con la mitad de tamaño. Con delegado GPU / Core ML. |
| `eggs_dano_v3_fp32.tflite` | 9,6 MB | Referencia. Útil si el delegado da problemas con FP16. |

Misma entrada y salida en los dos. En Drive: `MyDrive/eggs_v2/exports/v3_dano/`.

## Flujo

```
frame ──► v2 (cuadrado 640×640) ──► cajas + Crack/Intact
                                         │  solo huevos Crack
                                         ▼
          recorte del huevo en el FRAME COMPLETO (caja + 10 %) ──► 192×192 ──► v3_dano ──► máscaras
```

Recortar del **frame completo**, no del cuadrado de 640, aprovecha la resolución de la cámara, y así las grietas finas se ven mejor.

## Entrada

| | |
|---|---|
| forma | `[1, 192, 192, 3]` (NHWC) |
| tipo | `float32` |
| color | **RGB**, rango `0.0 – 1.0` (igual que `v2`) |

**Cómo se hace el recorte** (tiene que ser igual al del entrenamiento):
1. Tomar la caja de `v2` en píxeles del frame: `x1, y1, x2, y2`, con ancho `w` y alto `h`.
2. Expandirla un **10 %**: 5 % por cada lado, en ancho y en alto.
   `x1' = x1 − 0.05·w`, `x2' = x2 + 0.05·w`, `y1' = y1 − 0.05·h`, `y2' = y2 + 0.05·h`.
3. Recortar al borde del frame si se sale.
4. **Estirar** ese rectángulo a 192×192. Sin mantener la proporción: el modelo se entrenó con el recorte estirado.

## Salida

| | |
|---|---|
| forma | `[1, 192, 192, 2]` |
| tipo | `float32`, probabilidades `0–1` (sigmoide aplicada) |

En el `Float32Array` plano, el píxel `(x, y)` del canal `c` está en `out[(y * 192 + x) * 2 + c]`:

| canal `c` | contenido |
|---|---|
| 0 | silueta del **huevo** |
| 1 | **zona dañada** |

Un píxel es huevo o daño si su valor es `> 0.5`.

**Gravedad:** `daño / huevo`, contando solo los píxeles de daño que están **dentro** de la silueta.

| gravedad | % dañado (según el modelo) |
|---|---|
| leve | < 15 % |
| media | 15 – 35 % |
| grave | ≥ 35 % |

Los umbrales están pensados para la escala que da el modelo, que tiende a quedarse **por debajo** de la anotación (ver *Rendimiento*). Se pueden ajustar.

**Dibujar:** pinten solo el canal de daño. El canal de huevo puede dejar manchas pequeñas en el fondo del recorte, así que no dibujen su contorno. Para llevar la máscara al frame, el píxel `(x, y)` del recorte corresponde a `(rx + x · rw/192, ry + y · rh/192)`, donde `rx, ry, rw, rh` es el rectángulo recortado en el paso 3.

## Ejemplo (frame processor)

Continúa el ejemplo de `v2`: `eggs` son las detecciones ya pasadas por NMS, con la caja normalizada al cuadrado `crop` de lado `side`.

```ts
const S = 192;
const damageTflite = useTensorflowModel(require('../assets/eggs_dano_v3_fp16.tflite'), 'android-gpu'); // iOS: 'core-ml'
const damageModel = damageTflite.state === 'loaded' ? damageTflite.model : undefined;

// dentro del frame processor, después de obtener `eggs`:
for (const egg of eggs) {
  if (egg.cls !== 0 || damageModel == null) continue;          // solo huevos Crack
  // caja en píxeles del frame
  const x1 = crop.x + egg.box[0] * side, y1 = crop.y + egg.box[1] * side;
  const x2 = crop.x + egg.box[2] * side, y2 = crop.y + egg.box[3] * side;
  const w = x2 - x1, h = y2 - y1;
  // +10 % (5 % por lado), recortado al frame
  const rx = Math.max(0, x1 - 0.05 * w), ry = Math.max(0, y1 - 0.05 * h);
  const rw = Math.min(frame.width, x2 + 0.05 * w) - rx, rh = Math.min(frame.height, y2 + 0.05 * h) - ry;

  const input = resize(frame, {
    crop: { x: rx, y: ry, width: rw, height: rh },
    scale: { width: S, height: S },          // estirado, sin mantener proporción
    pixelFormat: 'rgb',
    dataType: 'float32',
  });
  const out = damageModel.runSync([input])[0] as Float32Array;

  let eggPx = 0, dmgPx = 0;
  const mask = new Uint8Array(48 * 48);      // máscara reducida para dibujar en el hilo JS
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = (y * S + x) * 2;
      if (out[i] > 0.5) {
        eggPx++;
        if (out[i + 1] > 0.5) {
          dmgPx++;
          mask[(y >> 2) * 48 + (x >> 2)] = 1;
        }
      }
    }
  }
  const severity = eggPx > 0 ? dmgPx / eggPx : 0;           // 0..1
  const level = severity < 0.15 ? 'leve' : severity < 0.35 ? 'media' : 'grave';
  // enviar { rect: [rx, ry, rw, rh], mask, severity, level } al hilo JS y dibujar
  // cada celda de `mask` como un rectángulo semitransparente de rw/48 × rh/48
}
```

## Recomendaciones

- **Solo sobre huevos Crack** y a menor frecuencia que el detector: 3–5 veces por segundo por huevo es suficiente. La máscara se puede reutilizar entre frames mientras la caja no se mueva mucho.
- **Suavizar la gravedad:** mostrar la media de los últimos 5–10 valores, igual que la clase.
- **Grietas del otro lado:** el modelo solo ve la cara que mira a la cámara. Girar el huevo cambia la gravedad; conviene mostrar el máximo observado mientras el huevo esté a la vista.

## Rendimiento medido

Datos: 35 huevos rajados y 30 sanos de **test**, que el modelo no vio al entrenar. **Tubería completa**: `v2` detecta la caja (encontró los 65 huevos, IoU medio de caja 0.88), se recorta como en la app y se segmenta.

| métrica | valor |
|---|---|
| IoU de la silueta del huevo | 0.90 |
| IoU de la zona dañada (huevos rajados) | 0.64 |
| Error medio de la gravedad | ±9,6 puntos de % |
| Huevos rajados con daño detectado (> 3 %) | 35 / 35 |
| Huevos sanos con daño detectado (> 3 %) | 0 / 30 |

`.tflite` frente a Keras: FP32 da exactamente lo mismo (diferencia < 1e-4). FP16 difiere hasta 0.23 en algún píxel suelto, pero las métricas son iguales (IoU de daño 0.641 frente a 0.641). Detalle por huevo en [`resultados/v3_dano/resumen.json`](resultados/v3_dano/resumen.json).

La gravedad predicha suele quedar **por debajo** de la anotada. Las zonas anotadas se marcaron sobre una rejilla de 10×10 y son algo más amplias que el daño real, así que el modelo es más ajustado que la etiqueta.

## Limitaciones

- La **zona dañada** es una región, no el trazo exacto de la grieta. Así se anotó: 420 huevos rajados: 351 de train, 34 de valid y 35 de test, sobre una rejilla de 10×10 por huevo.
- Las fotos del **montaje** (224 px) no se anotaron, porque la grieta casi nunca se distingue. En huevos de ese montaje el modelo puede marcar poco o ningún daño.
- Los negativos son huevos sanos del montaje. **No se ha validado con video real.** Es lo primero que hay que probar en la app.
