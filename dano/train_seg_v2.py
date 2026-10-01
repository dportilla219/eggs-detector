# Reentrena el segmentador de zona dañada con los recortes originales + los de v5 (colección del profesor,
# manchas sintéticas y huevos sanos). Parte de un .keras anterior si se da; si no, de ImageNet como train_seg.py.
# Guarda el mejor por la media del IoU de daño en el valid original y en el valid de v5.
import os, sys, json, numpy as np, tensorflow as tf
from segcommon import S, load_split, crop, augment

RUN, EPOCHS, V5 = sys.argv[1], int(sys.argv[2]), sys.argv[3]
INIT = sys.argv[4] if len(sys.argv) > 4 and os.path.exists(sys.argv[4]) else None
RUNS = os.environ.get('RUNS_DIR', 'runs')
os.makedirs(f'{RUNS}/{RUN}', exist_ok=False)   # nunca sobrescribir un run
tf.keras.utils.set_random_seed(0)

tr, va = load_split('train'), load_split('valid')
n5 = np.load(f'{V5}/train/datos.npz'); x5, y5 = n5['x'], n5['y']
v5 = np.load(f'{V5}/valid/datos.npz')
v5_x = v5['x'].astype(np.float32) / 255.; v5_y = (v5['y'] > 127).astype(np.float32)
v5_pos = v5_y[..., 1].sum((1, 2)) > 0
rng = np.random.default_rng(0)
print(f'original: {len(tr)} train, {len(va)} valid | v5: {len(x5)} train, {len(v5_x)} valid ({int(v5_pos.sum())} con daño) | inicio: {INIT or "ImageNet"}', flush=True)


def gen():
    while True:
        for i in rng.permutation(len(tr)):
            im, m = crop(tr[i], jitter=.08, rng=rng)     # uno original...
            im, m = augment(im, m, rng)
            yield im.astype(np.float32) / 255., m
            j = rng.integers(len(x5))                    # ...y uno de v5
            im, m = augment(x5[j], (y5[j] > 127).astype(np.float32), rng)
            yield im.astype(np.float32) / 255., m

va_x = np.stack([crop(it)[0] for it in va]).astype(np.float32) / 255.
va_y = np.stack([crop(it)[1] for it in va])
va_pos = np.array([it['pos'] for it in va])


def build():
    inp = tf.keras.Input((S, S, 3), name='image')                    # RGB 0..1
    x = tf.keras.layers.Rescaling(2., -1.)(inp)                      # MobileNetV2 espera -1..1
    base = tf.keras.applications.MobileNetV2((S, S, 3), alpha=.5, include_top=False, weights='imagenet')
    names = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu']
    enc = tf.keras.Model(base.input, [base.get_layer(n).output for n in names])
    skips = enc(x); x = skips[-1]
    for s, f in zip(reversed(skips[:-1]), [96, 64, 48, 32]):
        x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
        x = tf.keras.layers.Concatenate()([x, s])
        for _ in range(2):
            x = tf.keras.layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU(6.)(x)
    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(24, 3, padding='same', activation='relu')(x)
    out = tf.keras.layers.Conv2D(2, 1, activation='sigmoid', name='masks')(x)   # [huevo, daño]
    return tf.keras.Model(inp, out)


def loss(y, p):
    bce = tf.keras.losses.binary_crossentropy(y[..., None], p[..., None])
    bce = tf.reduce_mean(bce, [0, 1, 2]); inter = tf.reduce_sum(y * p, [1, 2])
    dice = 1 - (2 * inter + 1) / (tf.reduce_sum(y, [1, 2]) + tf.reduce_sum(p, [1, 2]) + 1)
    dice = tf.reduce_mean(dice, 0)
    return bce[0] + dice[0] + 2 * (bce[1] + dice[1])


def evaluate(model, x, y, pos):
    p = model.predict(x, batch_size=16, verbose=0) > .5; y = y > .5
    iou = lambda a, b: (a & b).sum((1, 2)) / np.maximum(1, (a | b).sum((1, 2)))
    egg = iou(p[..., 0], y[..., 0]); dmg = iou(p[..., 1], y[..., 1])
    frac_p = p[..., 1].sum((1, 2)) / np.maximum(1, p[..., 0].sum((1, 2)))
    frac_y = y[..., 1].sum((1, 2)) / np.maximum(1, y[..., 0].sum((1, 2)))
    return dict(egg_iou=float(egg.mean()), dmg_iou_pos=float(dmg[pos].mean()),
                sev_mae_pos=float(np.abs(frac_p - frac_y)[pos].mean()),
                neg_fp=float((frac_p[~pos] > .03).mean()) if (~pos).any() else 0.,
                pos_detected=float((frac_p[pos] > .03).mean()))


model = tf.keras.models.load_model(INIT, compile=False) if INIT else build()
if INIT:
    print('antes de reentrenar:', json.dumps({'orig': evaluate(model, va_x, va_y, va_pos), 'v5': evaluate(model, v5_x, v5_y, v5_pos)}), flush=True)
steps = 3 * len(tr) // 16
sched = tf.keras.optimizers.schedules.CosineDecay(3e-4 if INIT else 1e-3, EPOCHS * steps)
model.compile(tf.keras.optimizers.Adam(sched), loss)
ds = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec((S, S, 3), tf.float32), tf.TensorSpec((S, S, 2), tf.float32))).batch(16).prefetch(4)

best, hist = -1, []
for ep in range(EPOCHS):
    h = model.fit(ds, steps_per_epoch=steps, epochs=1, verbose=0)
    a, b = evaluate(model, va_x, va_y, va_pos), evaluate(model, v5_x, v5_y, v5_pos)
    m = {'epoch': ep, 'loss': float(h.history['loss'][0]), **{f'orig_{k}': v for k, v in a.items()}, **{f'v5_{k}': v for k, v in b.items()}}
    m['score'] = (a['dmg_iou_pos'] + b['dmg_iou_pos']) / 2
    hist.append(m); flag = ''
    if m['score'] > best:
        best = m['score']; model.save(f'{RUNS}/{RUN}/best.keras'); flag = ' *'
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}) + flag, flush=True)
    json.dump(hist, open(f'{RUNS}/{RUN}/hist.json', 'w'), indent=1)
model.save(f'{RUNS}/{RUN}/last.keras')
