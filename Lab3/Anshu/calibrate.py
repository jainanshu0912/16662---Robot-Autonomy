"""
Measure where the blocks really are, using the robot itself.

For each reference block you hand-guide the gripper (guide mode) so that the fingers are
centred on the block at grasp height (fingertips around the middle of the block, as if about
to grasp it), then press Enter. The end-effector position is recorded.

Reference points (8 in total):
  grid corners  g00 (far-left), g02 (far-right), g20 (near-left), g22 (near-right)
  home rows     hr0 + hr3 (red, first & last),  hb0 + hb3 (blue, first & last)
"far" = away from the robot, "left" = robot's left (+y).

Every other location is interpolated from these, so the grid may be slightly rotated or
unevenly spaced. Results go to calibration.json, which Lab3.py loads automatically.
Afterwards re-run `python Lab3.py --headless` to regenerate the trajectories.

You can also write calibration.json by hand: {"g00": [x, y, z], ...} in metres, robot base frame.
"""
import json

import numpy as np

POINTS = [
    ("g00", "grid corner: FAR row, robot's LEFT"),
    ("g02", "grid corner: FAR row, robot's RIGHT"),
    ("g20", "grid corner: NEAR row, robot's LEFT"),
    ("g22", "grid corner: NEAR row, robot's RIGHT"),
    ("hr0", "red home slot nearest the robot"),
    ("hr3", "red home slot furthest from the robot"),
    ("hb0", "blue home slot nearest the robot"),
    ("hb3", "blue home slot furthest from the robot"),
]


def main():
    from frankapy import FrankaArm
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()

    result = {}
    for name, desc in POINTS:
        print(f"\n[{name}] Put a block on the {desc}.")
        input("Press Enter to start guide mode, then move the gripper onto the block ...")
        fa.run_guide_mode(duration=60, block=False)
        input("Fingers centred around the block at grasp height? Press Enter to record ...")
        fa.stop_skill()
        p = fa.get_pose().translation
        result[name] = [float(v) for v in p]
        print(f"  recorded {name}: x={p[0]:.4f}  y={p[1]:.4f}  z={p[2]:.4f}")
        up = fa.get_pose()
        up.translation = up.translation + np.array([0.0, 0.0, 0.10])   # lift 10 cm straight up
        fa.goto_pose(up, duration=3.0)

    with open("calibration.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved calibration.json. Now run:  python Lab3.py --headless")
    fa.reset_joints()


if __name__ == "__main__":
    main()
