"""Round trip: place a box at a known position, project it, then recover the
position from the 2D box alone. If the geometry is right this is exact."""
import numpy as np
from src3.deep3dbox import fit_box, fit_box_search_yaw, project_box

K = np.array([[1266.0, 0, 816.0], [0, 1266.0, 491.0], [0, 0, 1.0]])
WLH = (1.95, 4.63, 1.74)

print("KNOWN yaw supplied:")
print(f"{'true position':>24s} {'yaw':>7s} {'recovered':>24s} {'error':>8s}")
for T, yaw in [((0., 1.5, 12.), 0.0), ((-3., 1.5, 20.), 0.4),
               ((4., 1.5, 30.), -1.2), ((0., 1.5, 45.), 1.57), ((-8., 1.5, 55.), 2.4)]:
    uv = project_box(np.array(T), WLH, yaw, K)
    box2d = (uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max())
    got, err = fit_box(box2d, WLH, yaw, K)
    if got is None:
        print("   FAILED"); continue
    d = float(np.linalg.norm(got - np.array(T)))
    print(f"  ({T[0]:6.1f},{T[1]:4.1f},{T[2]:5.1f}) {yaw:7.2f} "
          f"({got[0]:6.1f},{got[1]:4.1f},{got[2]:5.1f}) {d:7.3f}m")

print()
print("yaw SEARCHED (unknown orientation):")
print(f"{'true position':>24s} {'true yaw':>9s} {'recovered':>24s} {'yaw found':>10s} {'pos err':>8s}")
for T, yaw in [((0., 1.5, 12.), 0.0), ((-3., 1.5, 20.), 0.4), ((4., 1.5, 30.), -1.2)]:
    uv = project_box(np.array(T), WLH, yaw, K)
    box2d = (uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max())
    got, gyaw, err = fit_box_search_yaw(box2d, WLH, K)
    if got is None:
        print("   FAILED"); continue
    d = float(np.linalg.norm(got - np.array(T)))
    print(f"  ({T[0]:6.1f},{T[1]:4.1f},{T[2]:5.1f}) {yaw:9.2f} "
          f"({got[0]:6.1f},{got[1]:4.1f},{got[2]:5.1f}) {gyaw:10.2f} {d:7.3f}m")
