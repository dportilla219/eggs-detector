# Hoja de revisión: recorte original | silueta SAM (verde) + zona dañada anotada (magenta)
import glob, os, cv2, numpy as np, random, sys
SEG = os.environ.get('SEG_DIR', 'seg')
random.seed(int(sys.argv[2])); fs = random.sample(sorted(glob.glob(f'{SEG}/{sys.argv[1]}/*_p.npz')), 12)
T = []
for f in fs:
    d = np.load(f); im = cv2.imdecode(d['img'], 1); x1, y1, x2, y2 = d['box'].astype(int)
    v = im.copy(); ov = v.copy(); ov[d['dmg'] > 0] = (255, 0, 255); v = cv2.addWeighted(v, .5, ov, .5, 0)
    cn, _ = cv2.findContours(d['egg'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v, cn, -1, (0, 255, 0), 1)
    a = np.hstack([im[y1:y2, x1:x2], v[y1:y2, x1:x2]]); s = 440 / max(a.shape[:2]); a = cv2.resize(a, None, fx=s, fy=s)
    t = np.zeros((230, 450, 3), np.uint8); t[:min(230, a.shape[0]), :a.shape[1]] = a[:230]; T.append(t)
cv2.imwrite(sys.argv[3], np.vstack([np.hstack(T[i:i + 3]) for i in range(0, 12, 3)]))
