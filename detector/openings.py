"""Connect proximal wall crossings before merging overlapping traces."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

class WallOpenings:
    def __init__(self, ct, aorta, threshold):
        # Six-connected exterior wall voxels; no parent voxels in openings.
        wall=ndi.binary_dilation(aorta)&~aorta&(ct.data>=threshold)
        labels,_=ndi.label(wall,structure=np.ones((3,3,3),dtype=bool))
        coords=np.argwhere(wall)
        self.tree=cKDTree(ct.index_to_physical(coords)) if len(coords) else None
        self.labels=labels[tuple(coords.T)]
        self.tolerance=float(np.linalg.norm(ct.spacing))
    def associate(self, branch):
        if self.tree is None:return []
        distance,index=self.tree.query(branch['ostium_xyz_mm'])
        return [int(self.labels[index])] if distance<=self.tolerance else []

def same_daughter(a,b):
    if not set(a['wall_openings']).intersection(b['wall_openings']):return False
    if np.dot(a['direction_xyz'],b['direction_xyz'])<.65:return False
    radius=min(a['radius_mm'],b['radius_mm'])
    if np.linalg.norm(np.array(a['ostium_xyz_mm'])-b['ostium_xyz_mm'])>max(2.7,2*radius):return False
    def proximal(branch):
        path=np.array(branch['path_xyz_mm']);arc=np.array(branch['path_arc_mm'])
        return np.array([np.interp(np.linspace(0,5,21),arc,path[:,k]) for k in range(3)]).T
    aa,bb=proximal(a),proximal(b)
    distance=np.linalg.norm(aa[:,None]-bb[None],axis=-1)
    return max(np.median(distance.min(axis=0)),np.median(distance.min(axis=1)))<max(1.,radius)
