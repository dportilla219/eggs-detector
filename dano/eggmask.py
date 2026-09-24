# Cachea máscaras de huevo SAM (una por caja) en masks/<split>/<stem>.npz
import glob,cv2,numpy as np,os,sys
from ultralytics import SAM
sam=SAM('sam2.1_b.pt')
split=sys.argv[1]; lim=int(sys.argv[2]) if len(sys.argv)>2 else 10**9
os.makedirs(f'masks/{split}',exist_ok=True)
files=sorted(glob.glob(f'data/{split}/labels/*.txt'))[:lim]
if len(sys.argv)>3:
    keep=set(l.split()[0] for l in open(sys.argv[3]).read().splitlines() if l.strip())
    files=[f for f in files if os.path.basename(f)[:-4] in keep]
for i,f in enumerate(files):
    stem=os.path.basename(f)[:-4]; out=f'masks/{split}/{stem}.npz'
    if os.path.exists(out): continue
    ip=glob.glob(f'data/{split}/images/{stem}.*')[0]
    im=cv2.imread(ip);h,w=im.shape[:2]
    rows=[list(map(float,l.split())) for l in open(f) if l.strip()]
    boxes=[[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h] for c,cx,cy,bw,bh in rows]
    if boxes:
        r=sam(im,bboxes=boxes,verbose=False)[0]
        m=(r.masks.data.cpu().numpy()>0.5)
    else: m=np.zeros((0,h,w),bool)
    np.savez_compressed(out,m=np.packbits(m,axis=-1),shape=np.array(m.shape),cls=np.array([r_[0] for r_ in rows]))
    if i%100==0: print(split,i,len(files),flush=True)
