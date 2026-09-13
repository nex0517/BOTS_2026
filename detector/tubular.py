"""Local multi-scale contrast curvature along a traced vessel."""
import numpy as np
from .tracking import sample, point_at_arc, _basis


def evidence(ct,aorta,branch,parent_hu,background_hu):
    path=np.array(branch['path_xyz_mm']);arc=np.array(branch['path_arc_mm'])
    distances=np.linspace(2.5,min(8.,arc[-1]),4)
    centers=[];directions=[];rings=[];axial=[]
    for distance in distances:
        center=point_at_arc(path,arc,distance)
        direction=point_at_arc(path,arc,min(distance+.5,arc[-1]))-point_at_arc(path,arc,max(0,distance-.5))
        direction/=np.linalg.norm(direction);u,v=_basis(direction)
        plane=np.array([np.cos(a)*u+np.sin(a)*v for a in np.arange(8)*np.pi/4])
        for scale in [.8,1.5,2.5,4.]:
            centers.append(center);rings.append(center+scale*plane);axial.append(center+scale*np.array([direction,-direction]))
    centers=np.array(centers);rings=np.array(rings);axial=np.array(axial)
    middle=sample(ct,centers);radial=sample(ct,rings).reshape(-1,8)
    axial_values=sample(ct,axial).reshape(-1,2)
    span=max(10.,parent_hu-background_hu)
    transverse=np.maximum(0,middle-np.median(radial,axis=1))/span
    longitudinal=np.abs(middle-np.mean(axial_values,axis=1))/span
    scores=np.clip(transverse-longitudinal,0,1).reshape(-1,4)
    return float(np.median(scores.max(axis=1)))
