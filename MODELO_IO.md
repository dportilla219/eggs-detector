# Modelo de huevos: entrada, salida e integración en la app

Guía para integrar el detector en la app React Native / Expo con `react-native-vision-camera` + `react-native-fast-tflite`.

> **Pendiente de completar:** el modelo definitivo, el tamaño de los archivos, las métricas y el umbral salen de `resumen.json` al terminar `eggs_v2.ipynb`. Los valores marcados con ⏳ se actualizan entonces. El formato de entrada/salida descrito aquí no cambia.

## Archivos

| archivo | tamaño | cuándo usarlo |
|---|---|---|
| `eggs_<modelo>_fp32.tflite` | ⏳ ~12 MB | Recomendado. Con delegado GPU (Android) o Core ML (iOS), que lo ejecutan en FP16. |
| `eggs_<modelo>_int8.tflite` | ⏳ ~3–4 MB | Si el GPU delegate no está disponible o se necesita menos tamaño. Corre en CPU. |

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
2. Se descartan los candidatos con confianza `< CONF` (**⏳ 0.5**, recomendado por la curva F1).
3. NMS **agnóstico a la clase** con IoU 0.5: ordenar por confianza y descartar las cajas que solapen más de 0.5 con una ya aceptada, sin importar su clase. Así un mismo huevo no sale a la vez como Crack e Intact.
4. Convertir a píxeles del cuadrado recortado: `x1 = (cx - w/2) * lado`, `y1 = (cy - h/2) * lado`, etc. Después sumar el offset del recorte para llevarlas al frame.

## Ejemplo (frame processor)

```ts
import { useFrameProcessor } from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useTensorflowModel } from 'react-native-fast-tflite';

const N = 8400;
const CONF = 0.5; // ⏳ confirmar con resumen.json
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
  const tflite = useTensorflowModel(require('../assets/eggs_model_fp32.tflite'), 'android-gpu'); // iOS: 'core-ml'
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

⏳ Se completa con `resumen.json`: métricas por clase en test, acierto sobre fondos distintos al del entrenamiento y comparación `.pt` vs `.tflite`.

## Limitaciones conocidas

- Todas las fotos reales de huevos sanos vienen de un único montaje (fondo gris, base negra). Para compensarlo se entrenó con imágenes sintéticas (huevos recortados sobre otros fondos), pero **el modelo no se ha validado aún con video real** en otros fondos e iluminaciones. Conviene probarlo pronto y, si falla, guardar esos frames para reentrenar.
- Entrenado sobre todo con un huevo por imagen. Con varios huevos juntos (por ejemplo en un cartón) puede rendir peor.
