"""
3D boxes from 2D detections by geometry (Deep3DBox, Mousavian et al. 2017),
the method implemented by the repo in 3d_bounding/.

The idea: a 3D box of known size and orientation projects to a footprint
whose extremes touch all four sides of the observed 2D detection box. That
tight-fit requirement gives four equations in which the object's position is
the only unknown -- so position falls out of a linear solve, with no depth
sensor, no stereo pair and no depth network.

Where each ingredient comes from here:
    2D box       YOLO                      (src2.detection)
    dimensions   nuScenes size statistics  (src2.config.CAR_SIZE_WLH)
    orientation  supplied, or searched by reprojection error
    position     SOLVED by this package

Only the reference repo's *network* is framework-bound (TensorFlow, weights
on Google Drive); it predicts dimensions and orientation. The geometry is
the method itself and needs nothing but numpy, so it is reimplemented here
and fed from sources this project already measures.
"""
