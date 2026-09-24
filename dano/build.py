# Construye seg/<split>/<n>.npz con: imagen completa (ruta), caja del huevo (px), máscara huevo y daño (recortadas a la caja expandida)
import glob,cv2,numpy as np,os,json
SEG=os.environ.get('SEG_DIR','seg')
from annsheet import crops
from annmask import parse,refine
def eggm(split,stem,k):
    d=np.load(f'masks/{split}/{stem}.npz'); sh=d['shape']
    return np.unpackbits(d['m'],axis=-1)[...,:sh[2]].astype(np.uint8)[k]
def run(split,itemsf,annf,out,neg=False):
    os.makedirs(out,exist_ok=True)
    items=[l.split() for l in open(itemsf).read().splitlines() if l.strip()]
    ann={}
    if annf:
        for line in open(annf).read().splitlines():
            if not line.strip(): continue
            if 'skip' in line: ann[int(line.split(':')[0])]=None; continue
            i,c=parse(line); ann[i]=c
    meta=[]
    for i,(stem,k) in enumerate(items):
        k=int(k)
        if not neg and ann.get(i) is None: continue
        ip=glob.glob(f'data/{split}/images/{stem}.*')[0]
        im=cv2.imread(ip); h,w=im.shape[:2]
        l=open(f'data/{split}/labels/{stem}.txt').read().splitlines()[k].split()
        c,cx,cy,bw,bh=map(float,l)
        box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
        egg=eggm(split,stem,k)
        dmg=np.zeros((h,w),np.uint8)
        if not neg:
            for kk,cc,b,cr in crops(split,stem):
                if kk!=k: continue
                x1,y1,x2,y2=b
                dmg[y1:y2,x1:x2]=refine(cr,ann[i],egg[y1:y2,x1:x2])
        np.savez_compressed(f'{out}/{split}_{i}_{"n" if neg else "p"}.npz',img=cv2.imencode('.png',im)[1],box=np.array(box,np.float32),egg=egg,dmg=dmg)
        meta.append((i,stem,k,float(dmg.sum())/max(1,egg.sum())))
    return meta
M={}
M['train']=run('train','items_train.txt','ann_train.txt',f'{SEG}/train')+run('train','items_neg.txt',None,f'{SEG}/train',neg=True)
M['valid']=run('valid','items_valid.txt','ann_valid.txt',f'{SEG}/valid')+run('valid','items_neg_valid.txt',None,f'{SEG}/valid',neg=True)
M['test']=run('test','items_test.txt','ann_test.txt',f'{SEG}/test')+run('test','items_neg_test.txt',None,f'{SEG}/test',neg=True)
json.dump(M,open(f'{SEG}/meta.json','w'))
for s,m in M.items():
    r=np.array([x[3] for x in m]); p=r[r>0]
    print(s,len(m),'pos',len(p),'ratio pcts',np.percentile(p,[10,25,50,75,90]).round(3))
