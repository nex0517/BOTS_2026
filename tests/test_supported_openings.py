"""Geometry failures that must never be repaired by inserting expected counts."""
import unittest
import numpy as np
from detector.branchseed import detect, Config
from detector.openings import same_daughter
from detector.parent_geometry import ParentEnds
from detector.tracking import validate_trace
from tools.vmr_synthetic import make

class SupportedOpeningTests(unittest.TestCase):
    def check_case(self,name,spacing,seed,count):
        ct,mask,origins=make(name,spacing,seed)
        result,diagnostics=detect(ct,mask)
        self.assertEqual(len(result['daughters']),count)
        for b in diagnostics['branches']:
            self.assertEqual(b['path_status'],'traced')
            self.assertTrue(b['wall_openings'])
            self.assertGreaterEqual(b['confidence'],Config().acceptance_score)
            validate_trace(b)
        return diagnostics

    def test_separate_close_openings_survive_spacing_and_rotation(self):
        for spacing in [(1,1,1),(.8,.8,2),(1.5,1.5,1.5)]:
            for seed in [47000,47001]:
                with self.subTest(spacing=spacing,seed=seed):
                    self.check_case('close_openings',spacing,seed,2)

    def test_common_trunk_is_one_origin(self):
        for spacing in [(1,1,1),(.8,.8,2),(1.5,1.5,1.5)]:
            self.check_case('common_trunk',spacing,47001,1)

    def test_small_anisotropic_and_faint_supported_tubes(self):
        self.check_case('small_2mm',(.8,.8,2),47001,1)
        self.check_case('faint_3mm',(1,1,1),47001,1)

    def test_dots_parallel_vessels_blobs_and_caps_are_not_daughters(self):
        for name in ['empty','parallel_vessel','blob_3mm','blob_4mm','blob_6mm','cropped_cap','dot_chain']:
            with self.subTest(name=name):
                self.check_case(name,(1,1,1),47001,0)

    def test_disconnected_openings_are_never_merged_by_proximity(self):
        a={'wall_openings':[1], 'direction_xyz':[1,0,0], 'ostium_xyz_mm':[0,0,0], 'radius_mm':2}
        b=dict(a,wall_openings=[2],ostium_xyz_mm=[0,.1,0])
        self.assertFalse(same_daughter(a,b))

    def test_offset_traces_of_one_connected_opening_are_deduplicated(self):
        a={'wall_openings':[1], 'direction_xyz':[1,0,0], 'ostium_xyz_mm':[0,0,0],
           'radius_mm':3., 'path_xyz_mm':[[0,0,0],[10,0,0]], 'path_arc_mm':[0,10]}
        b=dict(a,ostium_xyz_mm=[3,0,0],path_xyz_mm=[[3,0,0],[13,0,0]])
        self.assertTrue(same_daughter(a,b))
        b['wall_openings']=[2]
        self.assertFalse(same_daughter(a,b))

    def test_parent_end_geometry_distinguishes_cap_from_side_wall(self):
        ct,mask,_=make('empty',(1,1,1),47000)
        ends=ParentEnds(ct,mask.data>0)
        center,normal,tangent,radius,flat=ends.ends[0]
        cap=dict(ostium_xyz_mm=center,direction_xyz=tangent,origin_diameter_mm=2*radius)
        self.assertEqual(ends.reject_reason(cap),'crop_cap')
        side=dict(ostium_xyz_mm=center-normal*8+np.array([6,0,0]),direction_xyz=[1,0,0],origin_diameter_mm=2)
        self.assertIsNone(ends.reject_reason(side))

if __name__=='__main__':unittest.main()
