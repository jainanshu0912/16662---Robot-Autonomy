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

Before moving, every waypoint of every stage is checked with frankapy's own virtual-wall
check. If any waypoint would hit a wall, the offending moves are printed and nothing runs.

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


def load_stage(path):
    """Loads a stage saved by Lab3.py. Works for a normal .npz file and also for an .npz
    that got unpacked into a folder of .npy files (e.g. after copying it inside a zip)."""
    if os.path.isdir(path):
        d = {os.path.splitext(f)[0]: np.load(os.path.join(path, f))
             for f in os.listdir(path) if f.endswith(".npy")}
    else:
        z = np.load(path)
        d = {k: z[k] for k in z.files}
    if "waypoints" not in d or "durations" not in d:
        raise SystemExit(f"{path} does not contain waypoints/durations")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the commands without moving")
    ap.add_argument("--folder", default="waypoints")
    ap.add_argument("--check-only", action="store_true",
                    help="connect to the robot, run the virtual-wall check, and exit")
    ap.add_argument("--start", metavar="STAGE",
                    help="skip all stages before this one (e.g. --start box)")
    ap.add_argument("--from-move", type=int, default=1, metavar="K",
                    help="with --start: begin at move K of that stage (blocks must already be "
                         "where moves 1..K-1 left them)")
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
            # frankapy asserts type(duration) is float/int, so numpy floats must be converted
            arm.goto_joints([float(v) for v in q], duration=float(T))

    def set_gripper(closed):
        if arm is None:
            print("  close_gripper" if closed else "  open_gripper")
        elif closed:
            arm.close_gripper()
        else:
            arm.open_gripper()

    # reset_joints() goes to frankapy's own home pose; the trajectories start from HOME_Q
    move_arm(HOME_Q, 4.0)

    found = glob.glob(os.path.join(args.folder, "*.npz"))
    files = {os.path.splitext(os.path.basename(f))[0]: f for f in found}
    if not files:
        raise SystemExit(f"No trajectories in {args.folder}/ - run `python Lab3.py --headless` first.")

    def stage_rank(name):
        for i, s in enumerate(STAGES):
            if name.endswith(s):
                return (i, 0 if name.startswith("reset") else 1)
        return (99, 0)

    # ---- virtual-wall pre-check (uses frankapy's own check, so it matches goto_joints) ----
    order = sorted(files, key=stage_rank)
    if arm is not None:
        bad = []
        for name in order:
            d = load_stage(files[name])
            labels = d["labels"] if "labels" in d else ["?"] * len(d["waypoints"])
            tcp = d["tcp"] if "tcp" in d else np.full((len(d["waypoints"]), 3), np.nan)
            for i, wp in enumerate(d["waypoints"]):
                if arm.is_joints_in_collision_with_boxes([float(v) for v in wp[:7]]):
                    bad.append((name, i, str(labels[i]), tcp[i]))
        if bad:
            print("\nThese waypoints are inside frankapy's virtual walls:")
            for name, i, label, p in bad:
                print(f"  {name} waypoint {i}: {label}   tcp = {np.round(p, 3)}")
            raise SystemExit("Nothing was executed. Send this list so the plan can be adjusted.")
        print("Virtual-wall check passed for all waypoints.")
        if args.check_only:
            return

    if args.start:
        if args.start not in order:
            raise SystemExit(f"--start {args.start}: no such stage (have: {', '.join(order)})")
        order = order[order.index(args.start):]

    closed = False
    for si, name in enumerate(order):
        d = load_stage(files[name])
        W, D = d["waypoints"], d["durations"]
        labels = [str(l) for l in d["labels"]] if "labels" in d else [""] * len(W)
        first = 0
        if si == 0 and args.start and args.from_move > 1:
            tag = f"move {args.from_move}:"
            starts = [i for i, l in enumerate(labels) if l.startswith(tag)]
            if not starts:
                raise SystemExit(f"--from-move {args.from_move}: stage {name} has no such move")
            first = starts[0]
            W, D, labels = W[first:], D[first:], labels[first:]
        print(f"\n=== {name}: {len(W)} waypoints" + (f" (starting at {labels[0]})" if first else "") + " ===")
        if arm is not None:
            input("Press Enter to run this stage (hand on the e-stop) ...")
        last_label = None
        for i, (wp, T) in enumerate(zip(W, D)):
            if labels[i] != last_label:
                last_label = labels[i]
                print(f"  [{name}] {last_label}")
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
