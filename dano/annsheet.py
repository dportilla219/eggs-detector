# Hoja de anotación: por huevo [recorte | recorte con rejilla 10x10 filas A-J, cols 0-9]
import glob,cv2,numpy as np,os,sys,json
G=10; SZ=300
def crops(split,stem):
    im=cv2.imread(glob.glob(f'data/{split}/images/{stem}.*')[0]); h,w=im.shape[:2]; out=[]
    for k,l in enumerate(open(f'data/{split}/labels/{stem}.txt')):
        p=l.split()
        if not p: continue
        c,cx,cy,bw,bh=map(float,p)
        bw,bh=bw*1.1,bh*1.1
        x1,y1=max(0,int((cx-bw/2)*w)),max(0,int((cy-bh/2)*h)); x2,y2=min(w,int((cx+bw/2)*w)),min(h,int((cy+bh/2)*h))
        out.append((k,int(c),(x1,y1,x2,y2),im[y1:y2,x1:x2]))
    return out
def tile(crop,name):
    s=SZ/max(crop.shape[:2]); a=cv2.resize(crop,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC if s>1 else cv2.INTER_AREA)
    hh,ww=a.shape[:2]; b=a.copy()
    for i in range(1,G):
        cv2.line(b,(0,int(i*hh/G)),(ww,int(i*hh/G)),(0,255,255),1); cv2.line(b,(int(i*ww/G),0),(int(i*ww/G),hh),(0,255,255),1)
    t=np.full((SZ+44,2*SZ+70,3),40,np.uint8); ox=SZ+30
    t[22:22+hh,0:ww]=a; t[22:22+hh,ox:ox+ww]=b
    for i in range(G):
        for yy in (16,22+hh+16):
            cv2.putText(t,str(i),(ox+int((i+.3)*ww/G),yy),0,0.45,(0,255,255),1)
        for xx in (ox-16,ox+ww+4):
            cv2.putText(t,'ABCDEFGHIJ'[i],(xx,22+int((i+.7)*hh/G)),0,0.45,(0,255,255),1)
    cv2.putText(t,name,(2,14),0,0.5,(255,255,255),1)
    return t
if __name__=='__main__':
    split,listf,start,n,out=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
    items=[l.split() for l in open(listf).read().splitlines() if l.strip()][start:start+n]  # "stem k"
    tiles=[]
    for idx,(stem,k) in enumerate(items):
        for kk,c,box,cr in crops(split,stem):
            if kk==int(k): tiles.append(tile(cr,f'#{start+idx}'))
    while len(tiles)%2: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out,np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
