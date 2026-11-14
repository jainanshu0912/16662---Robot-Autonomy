"""
Send the saved Lab 3 joint trajectories to the real Panda.

!!! TEMPLATE - adapt it to the interface you used on the real robot in Lab 2 !!!
This version assumes frankapy (FrankaArm.goto_joints / goto_gripper). If your lab uses a
different interface, only `move_arm` and `move_gripper` below need to change.

Waypoint file format (written by Lab3.py): rows of [q1 ... q7, gripper], plus per-row durations.
  - if a row's gripper value differs from the current one, it is a gripper action (arm stays still)
  - otherwise the arm moves to q in the given duration
Gripper values are per-finger openings (sim convention), so the total width is 2x the value.

SAFETY: always run with a partner holding the e-stop. Run `python run_real_robot.py --dry-run`
first, and consider SPEED_SCALE > 1 for the first real runs.
"""
import argparse
import glob
import os

import numpy as np

SPEED_SCALE = 1.5      # >1 slows everything down
MIN_SEGMENT = 0.5      # seconds; no arm segment is sent shorter than this
STAGES = ["pickaxe", "axe", "box", "bow"]   # missing files (infeasible patterns) are skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the commands without moving")
    ap.add_argument("--folder", default="waypoints")
    args = ap.parse_args()

    fa = None
    if not args.dry_run:
        from frankapy import FrankaArm
        fa = FrankaArm()
        fa.reset_joints()
        fa.open_gripper()

    def move_arm(q, T):
        if fa is None:
            print(f"  arm  -> {np.round(q, 3)} in {T:.2f}s")
        else:
            fa.goto_joints(list(q), duration=T, ignore_virtual_walls=True)

    def move_gripper(g):
        width = 2.0 * g
        if fa is None:
            print(f"  grip -> width {width * 100:.1f} cm")
        elif g < 0.03:
            fa.goto_gripper(width, grasp=True, force=10.0)
        else:
            fa.goto_gripper(width)

    files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(args.folder, "*.npz"))}
    order = [f for f in sorted(files, key=lambda n: min((i for i, s in enumerate(STAGES) if s in n), default=99))]
    grip = 0.04
    for name in order:
        d = np.load(files[name])
        W, D = d["waypoints"], d["durations"]
        print(f"\n=== {name}: {len(W)} waypoints ===")
        if fa is not None:
            input("Press Enter to run this stage (hand on the e-stop) ...")
        for wp, T in zip(W, D):
            if wp[7] != grip:
                grip = wp[7]
                move_gripper(grip)
            else:
                move_arm(wp[:7], max(MIN_SEGMENT, T * SPEED_SCALE))

    if fa is not None:
        fa.reset_joints()


if __name__ == "__main__":
    main()
