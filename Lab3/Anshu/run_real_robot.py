"""
Send the saved Lab 3 joint trajectories to the real Panda.

Uses the same frankapy interface as our Lab 2 real-robot script (lab2_franka.py):
FrankaArm.reset_joints(), goto_joints(), open_gripper(), close_gripper().

Waypoint files (written by `python Lab3.py`): waypoints/<stage>.npz with
  waypoints: rows of [q1 ... q7, gripper]
  durations: seconds for each row
  - if a row's gripper value differs from the current one, it is a gripper action
    (open/close, arm stays still)
  - otherwise the arm moves to q in the given duration

SAFETY: always run with a partner holding the e-stop. Run with --dry-run first, and keep
SPEED_SCALE > 1 for the first real runs.
"""
import argparse
import glob
import os

import numpy as np

HOME_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])   # same start pose as Lab3.py / Lab 2
SPEED_SCALE = 2.0      # >1 slows everything down (planned durations x this)
MIN_SEGMENT = 1.0      # seconds; no arm segment is sent shorter than this
STAGES = ["pickaxe", "axe", "box", "bow"]   # missing files (infeasible patterns) are skipped
GRIP_THRESHOLD = 0.03  # gripper values below this mean "closed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the commands without moving")
    ap.add_argument("--folder", default="waypoints")
    args = ap.parse_args()

    arm = None
    if not args.dry_run:
        from frankapy import FrankaArm
        arm = FrankaArm()
        arm.reset_joints()
        arm.open_gripper()

    def move_arm(q, T):
        if arm is None:
            print(f"  goto_joints {np.round(q, 3)}  ({T:.1f}s)")
        else:
            arm.goto_joints(list(q), duration=T)

    def set_gripper(closed):
        if arm is None:
            print("  close_gripper" if closed else "  open_gripper")
        elif closed:
            arm.close_gripper()
        else:
            arm.open_gripper()

    # reset_joints() goes to frankapy's own home pose; the trajectories start from HOME_Q
    move_arm(HOME_Q, 4.0)

    files = {os.path.splitext(os.path.basename(f))[0]: f
             for f in glob.glob(os.path.join(args.folder, "*.npz"))}
    if not files:
        raise SystemExit(f"No trajectories in {args.folder}/ - run `python Lab3.py --headless` first.")

    def stage_rank(name):
        for i, s in enumerate(STAGES):
            if name.endswith(s):
                return (i, 0 if name.startswith("reset") else 1)
        return (99, 0)

    closed = False
    for name in sorted(files, key=stage_rank):
        d = np.load(files[name])
        W, D = d["waypoints"], d["durations"]
        print(f"\n=== {name}: {len(W)} waypoints ===")
        if arm is not None:
            input("Press Enter to run this stage (hand on the e-stop) ...")
        for wp, T in zip(W, D):
            want_closed = wp[7] < GRIP_THRESHOLD
            if want_closed != closed:
                closed = want_closed
                set_gripper(closed)
            else:
                move_arm(wp[:7], max(MIN_SEGMENT, T * SPEED_SCALE))

    set_gripper(False)
    move_arm(HOME_Q, 4.0)


if __name__ == "__main__":
    main()
