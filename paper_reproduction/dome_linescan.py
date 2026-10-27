"""Phase-object line scan across a neutral spherical Pt particle at the experimental optics, analysed with the two
trackers and the CoM (SI Figure S6).  ->  dome_linescan.json"""
import os
import sys, json, numpy as np, scipy.fft as sfft
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trackers import *
from skimage.registration import phase_cross_correlation
LAM=0.04175716056489365; SIG=1.0087066046262614e-3; ALPHA=18.7e-3; K_DET=0.18644e-3
L=LAM/K_DET; N=1120; DX=L/N
k=sfft.fftfreq(N,DX); KX,KY=np.meshgrid(k,k,indexing='xy')
x=np.arange(N)*DX; X,Y=np.meshgrid(x,x,indexing='xy')
def probe(x0,y0,tx=0,ty=0):
    kt=np.array([tx,ty])*K_DET/LAM
    A=np.clip((ALPHA/LAM-np.hypot(KX-kt[0],KY-kt[1]))/(1/L)+0.5,0,1)
    p=sfft.ifft2(A*np.exp(-2j*np.pi*(KX*x0+KY*y0))); return p/np.sqrt((abs(p)**2).sum())
def det(psi,psf=1.0):
    D=np.abs(sfft.fftshift(sfft.fft2(psi)))**2; c=N//2; D=D[c-128:c+128,c-128:c+128]
    return gaussian_filter(D,psf)/D.sum()*1.5e5
def dome_vproj(R, V0, xc, yc):
    r=np.hypot(X-xc,Y-yc); t=2*np.sqrt(np.clip(R**2-r**2,0,None)); return V0*t
def linescan(Vp, xc, yc, rmax=80.0, step=0.43, psf=1.0, ups=100):
    vac=det(probe(xc+150, yc+150), psf)
    Tb=build_template_edge([vac]); c0=np.array(com(vac)); e0=fit_ellipse(vac)
    ph=np.exp(1j*SIG*Vp); rs=np.arange(0,rmax+1e-6,step); out=[]
    for r0 in rs:
        acc=[]
        for tx,ty in ((0,0),(0.5,0.5),(0.25,0.75),(0.75,0.25)):
            D=det(probe(xc+r0,yc,tx,ty)*ph,psf); eb=get_disk_edge(D)
            sb,_,_=phase_cross_correlation(Tb,eb,upsample_factor=ups)
            acc.append([-sb[1]-tx, fit_ellipse(D,edge=eb)[0]-e0[0]-tx, com(D)[0]-c0[0]-tx])
        out.append(np.mean(acc,0))
    return rs, np.array(out)
if __name__=='__main__':
    V0=float(sys.argv[1]) if len(sys.argv)>1 else 25.0
    xc=yc=L/2; R=19.15
    Vd=dome_vproj(R,V0,xc,yc)
    res={}
    rs,prof=linescan(Vd,xc,yc)
    res['mip']=dict(r=rs.tolist(),prof=prof.tolist())
    json.dump(dict(V0=V0,R=R,res=res),open('dome_linescan.json','w'))
