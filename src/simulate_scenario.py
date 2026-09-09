"""
Validates collision_risk.py's TTC/CPA/risk-classification logic against
clean, noise-free ground truth from a SUMO traffic simulation, instead of
noisy vision-derived measurements.

SUMO gives us exact vehicle positions/speeds directly from its own physics
(no camera, no YOLO, no homography) -- this is the "assume radar-quality
input" track discussed in conversation. Reuses the actual pipeline pieces
(ConstantAccelerationKalmanFilter, compute_ttc, compute_cpa, classify_risk)
rather than reimplementing them, so this genuinely tests the same code the
video pipeline uses.

Scenario (first pass): ego travels at a constant speed toward a stationary
lead vehicle. Because we script both vehicles' motion ourselves via TraCI
(not SUMO's own car-following model), we know the exact ground truth at
every step and can compute the analytically-correct TTC independently:
given ego at constant speed v_ego and a stationary lead car, the true gap
closes linearly, so TTC(t) = (initial_gap - v_ego * t) / v_ego.
"""

import os
import sys
import time

import traci

SUMO_HOME = os.environ.get("SUMO_HOME")
if SUMO_HOME and os.path.join(SUMO_HOME, "tools") not in sys.path:
    sys.path.append(os.path.join(SUMO_HOME, "tools"))

sys.path.insert(0, os.path.dirname(__file__))
from collision_risk import classify_risk, compute_cpa, compute_ttc  # noqa: E402
from kalman_filter import ConstantAccelerationKalmanFilter  # noqa: E402

SUMO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "sumo", "scenario.sumocfg")
STEP_LENGTH = 0.1  # seconds -- matches the CCD dashcam clips' 10fps for comparability
GUI = True  # True = open the live sumo-gui window; False = run headless (faster, for repeated runs)

EGO_SPEED = 15.0       # m/s (~54 km/h), held constant throughout
INITIAL_GAP = 60.0     # meters, ego behind lead at simulation start
LEAD_SPEED = 0.0       # m/s -- stationary lead vehicle for this first pass


def analytical_ttc(initial_gap, ego_speed, t):
    """Exact TTC for a constant-speed ego closing on a stationary lead car."""
    remaining_gap = initial_gap - ego_speed * t
    if remaining_gap <= 0 or ego_speed <= 0:
        return 0.0
    return remaining_gap / ego_speed


def main():
    # sys.executable is the venv's python.exe; sumo(-gui).exe lives alongside
    # it in the same Scripts/ directory (both installed by the eclipse-sumo
    # pip package), so this finds it without relying on PATH.
    binary_name = "sumo-gui.exe" if GUI else "sumo.exe"
    sumo_binary = os.path.join(os.path.dirname(sys.executable), binary_name)
    sumo_cmd = [sumo_binary, "-c", SUMO_CONFIG, "--step-length", str(STEP_LENGTH)]
    if GUI:
        sumo_cmd += ["--start"]
    traci.start(sumo_cmd)

    # Place lead ahead of ego by INITIAL_GAP on the same straight edge/lane.
    traci.vehicle.add("ego", routeID="straight", typeID="car", departPos="0", departSpeed=str(EGO_SPEED))
    traci.vehicle.add("lead", routeID="straight", typeID="car", departPos=str(INITIAL_GAP), departSpeed=str(LEAD_SPEED))
    traci.vehicle.setSpeedMode("ego", 0)   # fully manual speed control, ignore SUMO's own safety/limit checks
    traci.vehicle.setSpeedMode("lead", 0)
    traci.simulationStep()  # let both vehicles actually enter the simulation

    kf = None
    t = 0.0
    step = 0

    while t < 8.0:
        traci.vehicle.setSpeed("ego", EGO_SPEED)
        traci.vehicle.setSpeed("lead", LEAD_SPEED)
        traci.simulationStep()
        step += 1
        t = step * STEP_LENGTH

        if GUI:
            # --delay only affects SUMO's own auto-play loop, not steps we
            # drive externally via TraCI -- without an actual real-time
            # pause here, all 35+ steps fire back-to-back in a fraction of
            # a second and the window is gone before you can see anything.
            time.sleep(0.15)

        if "ego" not in traci.vehicle.getIDList() or "lead" not in traci.vehicle.getIDList():
            break

        ego_x, ego_y = traci.vehicle.getPosition("ego")
        lead_x, lead_y = traci.vehicle.getPosition("lead")

        # Straight road along x: relative forward gap is z, no lateral offset.
        rel_x = lead_y - ego_y
        rel_z = lead_x - ego_x

        if kf is None:
            kf = ConstantAccelerationKalmanFilter(rel_x, rel_z, STEP_LENGTH)
        else:
            kf.predict()
            kf.update(rel_x, rel_z)

        x, z = kf.position
        vx, vz = kf.velocity

        ttc = compute_ttc(x, z, vx, vz)
        cpa_time, cpa_distance = compute_cpa(x, z, vx, vz)
        risk = classify_risk(ttc, cpa_time, cpa_distance)

        true_ttc = analytical_ttc(INITIAL_GAP, EGO_SPEED, t)
        ttc_str = f"{ttc:.2f}s" if ttc is not None else "N/A"

        print(f"t={t:4.1f}s  true_gap={INITIAL_GAP - EGO_SPEED*t:6.2f}m  "
              f"filtered_z={z:6.2f}m  filtered_vz={vz:6.2f}m/s  "
              f"our_ttc={ttc_str:>8}  true_ttc={true_ttc:6.2f}s  risk={risk}")

    if GUI:
        print("Simulation finished -- leaving the window open for 5s so you can see the final state.")
        time.sleep(5)

    traci.close()


if __name__ == "__main__":
    main()
