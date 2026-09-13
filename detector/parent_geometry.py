"""Parent centerline endpoints and local cap/continuation geometry."""
import numpy as np
from scipy import ndimage as ndi


class ParentEnds:
    def __init__(self,ct,aorta):
        coords=np.argwhere(aorta)
        axis=int(np.argmax(np.ptp(coords,axis=0)*ct.spacing))
        levels=np.arange(coords[:,axis].min(),coords[:,axis].max()+1)
        count=np.bincount(coords[:,axis]-levels[0],minlength=len(levels))
        centers=np.column_stack([np.bincount(coords[:,axis]-levels[0],weights=coords[:,k],minlength=len(levels))/np.maximum(1,count) for k in range(3)])
        valid=count>0
        for k in range(3):centers[:,k]=np.interp(levels,levels[valid],centers[valid,k])
        smooth=ndi.gaussian_filter1d(centers,2/ct.spacing[axis],axis=0,mode='nearest')
        points=ct.index_to_physical(smooth)
        step=min(len(points)-1,max(1,int(np.ceil(6/ct.spacing[axis]))))
        self.ends=[]
        normal=ct.affine[:3,axis]/ct.spacing[axis]
        area=float(np.prod(np.delete(ct.spacing,axis)))
        for end,neighbor,sign in [(0,step,-1),(-1,-1-step,1)]:
            tangent=points[end]-points[neighbor];tangent/=max(np.linalg.norm(tangent),1e-9)
            center=ct.index_to_physical(centers[end])[0]
            radius=np.sqrt(np.median(count[:step+1] if end==0 else count[-step-1:])*area/np.pi)
            flat=count[end]>=.5*max(1,count[neighbor])
            self.ends.append((center,normal*sign,tangent,radius,flat))
        self.margin=max(.75,ct.spacing[axis]/2)

    def reject_reason(self,branch):
        origin=np.array(branch['ostium_xyz_mm']);direction=np.array(branch['direction_xyz'])
        for center,normal,tangent,radius,flat in self.ends:
            inward=-np.dot(origin-center,normal)
            if flat and inward<self.margin:
                return 'crop_cap'
            near=np.linalg.norm(origin-center)<2*radius+self.margin
            if near and np.dot(direction,tangent)>.8 and branch['origin_diameter_mm']/2>.55*radius:
                return 'parent_continuation'
        return None
