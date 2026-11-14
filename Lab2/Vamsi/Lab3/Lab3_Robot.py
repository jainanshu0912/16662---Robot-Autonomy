"""
Lab 3 on the real Franka: same planning as Lab3_Sim.py (Lab3Runner in
Lab3_Shared.py), executed through frankapy instead of MuJoCo.

The robot is driven the same way as in lab2_franka.py: FrankaArm(),
reset_joints(), one goto_joints() per joint waypoint, open_gripper() and
close_gripper().

The robot runs OPEN LOOP: it assumes every block sits exactly at its nominal
location (see the geometry constants in SymScript_Lab3.py). Measure the grid
and slot positions on the real table before running.

SAFETY: never run alone; keep a hand on the emergency stop.

Suggested order on the robot
  1. python Lab3_Robot.py --dry-run                  # no robot: prints what would be sent
  2. python Lab3_Robot.py --z-offset 0.05 --no-gripper --step --patterns pickaxe
                                                      # no blocks, 5 cm high, confirm each action
  3. python Lab3_Robot.py --step --patterns pickaxe   # with blocks, one pattern
  4. python Lab3_Robot.py                             # full run for the video

Options
  --reset          return the blocks to the start layout between patterns
  --speed 0.5      scale on the planned joint speeds (1.0 = same as the sim)
"""
import argparse

import numpy as np

import SymScript_Lab3 as L
from Lab3_Shared import GRIPPER_OPEN, Q_HOME, Lab3Runner

# frankapy's home joints (FrankaArm.reset_joints); used as the start in --dry-run
FRANKAPY_HOME = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

MIN_MOVE_TIME = 1.5        # s, shortest duration sent to goto_joints
LINE_TOL = 0.002           # m, allowed TCP deviation from the straight approach line
START_TOL = 0.05           # rad, the robot must be this close to the start of a path


# ----------------------------------------------------------------------------- arms
class DryRunArm:
    """Stands in for frankapy.FrankaArm: records commands instead of moving."""

    def __init__(self, verbose=False):
        self.q = FRANKAPY_HOME.copy()
        self.verbose = verbose
        self.n_moves, self.n_grips, self.total_time = 0, 0, 0.0

    def reset_joints(self):
        self.q = FRANKAPY_HOME.copy()

    def get_joints(self):
        return self.q.copy()

    def goto_joints(self, joints, duration=5):
        self.q = np.array(joints, dtype=float)
        self.n_moves += 1
        self.total_time += duration
        if self.verbose:
            print("      goto_joints(%s, duration=%.1f)" % (np.round(self.q, 3).tolist(), duration))

    def open_gripper(self):
        self.n_grips += 1
        if self.verbose:
            print("      open_gripper()")

    def close_gripper(self):
        self.n_grips += 1
        if self.verbose:
            print("      close_gripper()")


def connect_robot():
    from frankapy import FrankaArm   # imported here so --dry-run works without frankapy
    return FrankaArm()


# ----------------------------------------------------------------------------- executor
class FrankaExecutor:
    """Executes Lab 2 style steps on a frankapy FrankaArm (or DryRunArm).

    Same interface as Lab 2's SimExecutor: Execute(steps).
      move steps    -> one goto_joints per waypoint
      gripper steps -> open_gripper / close_gripper
    """

    def __init__(self, arm, mplanner_getter, speed_scale=0.5, use_gripper=True):
        self.arm = arm
        self._mplanner = mplanner_getter   # callable: the runner creates the planner later
        self.speed_scale = speed_scale
        self.use_gripper = use_gripper
        self.max_line_dev = 0.0

    # --- straight-line approaches -------------------------------------------------
    def _line_deviation(self, qa, qb):
        """Largest distance of the TCP from the straight line TCP(qa)->TCP(qb) when the
        robot interpolates linearly in joint space (which is what goto_joints does)."""
        mp = self._mplanner()
        pa, pb = mp.TCPPose(qa)[:3, 3], mp.TCPPose(qb)[:3, 3]
        d = pb - pa
        n = np.linalg.norm(d)
        dev = 0.0
        for s in np.linspace(0, 1, 7)[1:-1]:
            p = mp.TCPPose(qa + s*(qb - qa))[:3, 3] - pa
            if n > 1e-9:
                p = p - (p @ d)/n**2 * d
            dev = max(dev, np.linalg.norm(p))
        return dev

    def _thin(self, path):
        """Drop intermediate waypoints of a straight-line (Cartesian) path as long as the
        TCP stays within LINE_TOL of the line, so the robot stops as rarely as possible."""
        path = [np.array(q, dtype=float) for q in path]
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and self._line_deviation(path[i], path[j]) > LINE_TOL:
                j -= 1
            self.max_line_dev = max(self.max_line_dev, self._line_deviation(path[i], path[j]))
            out.append(path[j])
            i = j
        return out

    # --- execution ------------------------------------------------------------------
    def _move(self, path, speed):
        path = [np.array(q, dtype=float) for q in path]
        if len(path) < 2:
            return
        err = np.max(np.abs(np.array(self.arm.get_joints()) - path[0]))
        if err > START_TOL:
            raise RuntimeError("robot is %.3f rad away from the start of the planned path; "
                               "stopping instead of jumping there" % err)
        if speed < 0.5:           # slow moves are the straight-line approach / retreat
            path = self._thin(path)
        for q_prev, q in zip(path[:-1], path[1:]):
            dq = np.max(np.abs(q - q_prev))
            # min-jerk timing as in Lab 2: peak joint speed = 1.875 * distance / duration
            duration = max(MIN_MOVE_TIME, 1.875*dq/(speed*self.speed_scale))
            self.arm.goto_joints(q.tolist(), duration=duration)

    def _gripper(self, value):
        if not self.use_gripper:
            return
        if value >= GRIPPER_OPEN:
            self.arm.open_gripper()
        else:
            self.arm.close_gripper()

    def Execute(self, steps):
        for st in steps:
            if st["type"] == "move":
                self._move(st["path"], st["speed"])
            else:
                self._gripper(st["value"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="do not connect to the robot; print what would be sent")
    ap.add_argument("--verbose", action="store_true", help="with --dry-run: print every command")
    ap.add_argument("--reset", action="store_true", help="reset to the start layout between patterns")
    ap.add_argument("--patterns", nargs="+", choices=L.PATTERN_ORDER, help="subset of patterns (default: all four)")
    ap.add_argument("--z-offset", type=float, default=0.0, help="raise every grasp/place height by this many metres")
    ap.add_argument("--speed", type=float, default=0.5, help="scale on planned joint speeds (default 0.5)")
    ap.add_argument("--no-gripper", action="store_true", help="never open/close the gripper")
    ap.add_argument("--step", action="store_true", help="wait for Enter before every pick/place")
    ap.add_argument("--yes", action="store_true", help="skip the start confirmation")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if args.dry_run:
        arm = DryRunArm(args.verbose)
    else:
        print("About to move the REAL robot.")
        print("  - a partner is present and has a hand on the emergency stop")
        print("  - blocks are at the start layout (red on the robot's left, blue on its right)")
        print("  - z offset %.3f m, speed scale %.2f, gripper %s"
              % (args.z_offset, args.speed, "OFF" if args.no_gripper else "on"))
        if not args.yes and input("Type 'go' to start: ").strip().lower() != "go":
            print("Aborted.")
            return
        arm = connect_robot()

    # Same start-up as lab2_franka.py: reset, then go to the Lab 2 HOME pose that the
    # simulator also starts from, so the robot runs the motions Lab3_Sim.py verified.
    arm.reset_joints()
    arm.goto_joints(Q_HOME.tolist())
    if not args.no_gripper:
        arm.open_gripper()
    q_start = np.array(arm.get_joints(), dtype=float)

    def before_action(act):
        if args.dry_run:
            return
        if args.step:
            if input("   next: %s  [Enter = go, q = quit] " % act.label()).strip().lower() == "q":
                raise KeyboardInterrupt
        else:
            print("   -> %s" % act.label())

    runner_box = {}
    ex = FrankaExecutor(arm, lambda: runner_box["r"].mplanner, speed_scale=args.speed,
                        use_gripper=not args.no_gripper)
    runner = Lab3Runner(ex, q_start=q_start, z_offset=args.z_offset, seed=args.seed,
                        before_action=before_action)
    runner_box["r"] = runner

    try:
        runner.run_all(patterns=args.patterns, reset=args.reset)
        runner.go_home()
    except KeyboardInterrupt:
        print("\nStopped by the operator. The robot is left where it is.")
        return

    runner.print_summary("sent to the robot" if not args.dry_run else "planned (dry run)")
    print("   largest deviation from a straight approach line: %.1f mm" % (1e3*ex.max_line_dev))
    if args.dry_run:
        print("   dry run: %d goto_joints, %d gripper commands, about %.0f min of motion"
              % (arm.n_moves, arm.n_grips, arm.total_time/60))


if __name__ == "__main__":
    main()
