"""Deterministic synthetic geometry/contrast perturbations, independent of VMR IDs."""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from detector.branchseed import Volume,detect
from detector.tracking import validate_trace

SCENARIOS=('empty','small_2mm','faint_3mm','close_openings','common_trunk','parallel_vessel','blob_3mm','blob_4mm','blob_6mm','oblique','curved','cropped_cap','dot_chain')

def make(name,spacing,seed):
 spacing=np.array(spacing);shape=np.ceil(np.array([64,64,96])/spacing).astype(int)
 xyz=np.indices(shape,dtype=np.float32)*spacing[:,None,None,None]
 center=np.array([32.,32.,48.])+(np.array([.27,.19,.31]) if seed%2 else 0)
 x,y,z=xyz;cx,cy,cz=center
 parent=((x-cx)**2+(y-cy)**2<=6**2)&(z>=12)&(z<=84)
 rng=np.random.default_rng(seed);ct=rng.normal(35,12,shape).astype(np.float32);ct[parent]=rng.normal(300,12,parent.sum())
 origins=[]
 def tube(start,end,radius,value=280):
  start=np.array(start);delta=np.array(end)-start
  rel=xyz-start[:,None,None,None]
  t=np.clip(np.sum(rel*delta[:,None,None,None],axis=0)/np.dot(delta,delta),0,1)
  region=np.sum((rel-t[None]*delta[:,None,None,None])**2,axis=0)<=radius**2
  ct[region]=value+rng.normal(0,8,region.sum())
 if name in ('small_2mm','faint_3mm','oblique','curved'):
  origin=center+[6,0,0];origins=[origin]
  if name=='oblique':
   direction=np.array([1.,.25,.35]);direction/=np.linalg.norm(direction);tube(origin-2*direction,origin+14*direction,1.5)
  elif name=='curved':
   points=[center+[4+t,.04*t*t,.08*t] for t in np.arange(0,16,1.)]
   for a,b in zip(points,points[1:]):tube(a,b,1.7)
  else:tube(center+[4,0,0],center+[20,0,0],1. if name=='small_2mm' else 1.5,145 if name=='faint_3mm' else 280)
 elif name=='close_openings':
  for offset in [-2.25,2.25]:
   origins.append(center+[6,0,offset]);tube(center+[4,0,offset],center+[20,0,offset],1.25)
 elif name=='common_trunk':
  origins=[center+[6,0,0]];tube(center+[4,0,0],center+[14,0,0],2.)
  for side in [-1,1]:tube(center+[14,0,0],center+[22,side*6,0],1.5)
 elif name=='parallel_vessel':tube(center+[8.75,0,-15],center+[8.75,0,15],1.25)
 elif name.startswith('blob_'):
  radius=float(name.split('_')[1].replace('mm',''));p=center+[6+radius-.75,0,0]
  region=np.sum((xyz-p[:,None,None,None])**2,axis=0)<=radius**2;ct[region]=280
 elif name=='cropped_cap':tube([cx,cy,82],[cx,cy,95],5.)
 elif name=='dot_chain':
  for xx in np.arange(cx+7,cx+22,2.5):
   p=np.array([xx,cy,cz]);region=np.sum((xyz-p[:,None,None,None])**2,axis=0)<=.7**2;ct[region]=320
 angle=.61 if seed%2 else 0.;tilt=.37 if seed%2 else 0.
 rz=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
 rx=np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]])
 rotation=rz@rx;affine=np.eye(4);affine[:3,:3]=rotation@np.diag(spacing);affine[:3,3]=[17,-91,203]
 refs=[p@rotation.T+affine[:3,3] for p in origins]
 return Volume(ct,affine,spacing),Volume(parent,affine,spacing),refs


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threshold',type=float,help='Optional additional score sensitivity cutoff')
    args=parser.parse_args()
    rows=[]
    for spacing in [(1.,1.,1.),(.8,.8,2.),(1.5,1.5,1.5)]:
     for name in SCENARIOS:
      for seed in [47000,47001]:
       ct,mask,refs=make(name,spacing,seed);start=time.perf_counter();result,diag=detect(ct,mask);elapsed=time.perf_counter()-start
       if args.threshold is not None:
        keep=[i for i,b in enumerate(diag['branches']) if b['confidence']>=args.threshold];result['daughters']=[result['daughters'][i] for i in keep];diag['branches']=[diag['branches'][i] for i in keep]
       ds=result['daughters'];dist=np.array([[np.linalg.norm(np.array(p['ostium_xyz_mm'])-r) for r in refs] for p in ds]).reshape(len(ds),len(refs));ii,jj=linear_sum_assignment(np.where(dist<=3,dist,1000));tp=sum(dist[i,j]<=3 for i,j in zip(ii,jj))
       for b in diag['branches']:
        if b.get('path_status')=='traced':validate_trace(b)
       rows.append(dict(scenario=name,spacing=list(spacing),seed=seed,expected=len(refs),predicted=len(ds),tp=int(tp),fp=len(ds)-int(tp),fn=len(refs)-int(tp),runtime_s=elapsed,unresolved=sum(b.get('path_status')!='traced' for b in diag['branches'])))
      print(name,spacing,[(x['tp'],x['fp'],x['fn']) for x in rows[-2:]],flush=True)
    Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')

if __name__=="__main__":
    main()
