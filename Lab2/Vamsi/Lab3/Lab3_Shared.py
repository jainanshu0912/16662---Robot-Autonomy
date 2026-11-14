"""
Code shared by the simulator (Lab3_Sim.py) and the real robot (Lab3_Robot.py).

  GraspLibrary / action_steps : turn one symbolic pick/place from
                                SymScript_Lab3 into Lab 2 style steps
  Lab3Runner                  : the whole lab loop (task plan -> motion plan ->
                                execute) for any executor

An executor is any object with Execute(steps); the sim uses Lab 2's
SimExecutor and the robot uses FrankaExecutor. Steps are

    {"type": "move", "path": [q, ...], "speed": rad/s}
    {"type": "gripper", "value": width}

using the Lab 2 MotionPlanner (IK, straight-line approach, RRT transit).
"""
import io
import os
import sys
import types
import contextlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_lab2():
    """Folder holding the Lab 2 stack (MotionPlanner.py, TaskPlanner.py, SimpleScript.py,
    Franka.py, RobotUtil.py, franka_emika_panda/). Checked in order: $LAB2_DIR, the
    folder containing this Lab3 folder, this folder, and the original HW3 layout."""
    candidates = [os.environ.get("LAB2_DIR", ""), os.path.join(HERE, ".."), HERE,
                  os.path.join(HERE, "..", "..", "Lab2", "Lab2")]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "MotionPlanner.py")) \
                and os.path.isdir(os.path.join(c, "franka_emika_panda")):
            return os.path.abspath(c)
    raise RuntimeError("Lab 2 files (MotionPlanner.py, franka_emika_panda/, ...) not found; set LAB2_DIR")


LAB2_DIR = _find_lab2()
sys.path.insert(0, LAB2_DIR)

try:
    import mujoco  # noqa: F401
except ImportError:
    # The robot PC may not have MuJoCo. Lab 2's SimpleScript imports it at the
    # top, but we only need its scene description, so a placeholder is enough.
    _mj = types.ModuleType("mujoco")
    _mj.viewer = types.ModuleType("mujoco.viewer")
    sys.modules["mujoco"], sys.modules["mujoco.viewer"] = _mj, _mj.viewer

import SymScript_Lab3 as L                                           # noqa: E402
from MotionPlanner import CollisionEnv, MotionPlanner, GraspFrame    # noqa: E402,F401
from TaskPlanner import GRIPPER_OPEN, GRIPPER_CLOSED, Offset, UP     # noqa: E402,F401
from SimpleScript import BLOCKS, PLANNING_ONLY_BLOCKS, Q_HOME        # noqa: E402,F401

CLEARANCE = 0.12        # pre-grasp height above the grasp pose
PLACE_DROP = 0.003      # release the block this far above the table

# Planning-only box covering every block location, so transit motions stay
# above the blocks. It is ignored during the straight-line approach/retreat.
BLOCK_ZONE = ["BlockZone", [L.GRID_CENTER[0], 0.0, 0.03], [0.19, 0.20, 0.03]]
TOUCHABLE = ["TablePlane", "BlockZone"]


def make_motion_planner(seed=13):
    return MotionPlanner(CollisionEnv(BLOCKS + PLANNING_ONLY_BLOCKS + [BLOCK_ZONE], margin=0.01), seed=seed)


class GraspLibrary:
    """Pre-grasp configuration and straight-line approach for each (location, axis, kind).

    Every location is visited many times, so each approach is planned once
    (IK seeded from home) and reused; the retreat is the approach reversed."""

    def __init__(self, planner, z_offset=0.0):
        self.planner = planner
        self.z_offset = z_offset   # raise every grasp/place height (dry runs on the robot)
        self.cache = {}

    def get(self, loc, axis, kind):
        key = (loc, axis, kind)
        if key not in self.cache:
            self.cache[key] = self._plan(loc, axis, kind)
        return self.cache[key]

    def _plan(self, loc, axis, kind):
        pos = np.array(L.location_xyz(loc)) + (PLACE_DROP*UP if kind == "place" else 0) + self.z_offset*UP
        f = np.array(L.AXIS_FINGER_DIR[axis])
        best = None
        for finger in (f, -f):  # the gripper is symmetric: two equivalent hand yaws
            T = GraspFrame(pos, -UP, finger)
            for q_pre in self.planner.IKSolutions(Offset(T, CLEARANCE*UP), [Q_HOME], n_random=4):
                try:
                    approach = self.planner.CartesianPath(q_pre, T, ignore=TOUCHABLE)
                except RuntimeError:
                    continue
                cost = np.linalg.norm(q_pre - Q_HOME)
                if best is None or cost < best[0]:
                    best = (cost, approach)
                break  # solutions are sorted by distance to home
        if best is None:
            raise RuntimeError("no grasp found for %s (%s, %s)" % (loc, axis, kind))
        return best[1]


def action_steps(planner, grasps, q_now, act):
    """Lab 2 style steps (move / gripper dicts) for one symbolic pick or place."""
    loc, _, axis = act.args
    approach = grasps.get(loc, axis, act.name)
    steps = [
        {"type": "move", "path": planner.PlanToConfig(q_now, approach[0]), "speed": 0.8},
        {"type": "move", "path": approach, "speed": 0.25},
        {"type": "gripper", "value": GRIPPER_CLOSED if act.name == "pick" else GRIPPER_OPEN},
        {"type": "move", "path": approach[::-1], "speed": 0.25},
    ]
    return steps, approach[0]


class Lab3Runner:
    """Runs the lab on any executor: for each pattern, task-plan with
    SymScript_Lab3, motion-plan every pick/place and hand the steps to
    executor.Execute(). Infeasible patterns are skipped.

    Optional hooks:
      before_action(act)      called before an action is executed
      after_action(act)       called after an action is executed
      after_pattern(name)     called after a pattern has been built
    """

    def __init__(self, executor, q_start=Q_HOME, z_offset=0.0, seed=13,
                 before_action=None, after_action=None, after_pattern=None):
        self.executor = executor
        self.q_home = np.array(q_start, dtype=float)
        self.q_now = self.q_home.copy()
        self.mplanner = make_motion_planner(seed)
        self.grasps = GraspLibrary(self.mplanner, z_offset)
        self.tplanner = L.Lab3Planner(L.make_Lab3_domain())
        self.statics = L.static_facts()
        self.state = L.start_layout()
        self.before_action = before_action or (lambda act: None)
        self.after_action = after_action or (lambda act: None)
        self.after_pattern = after_pattern or (lambda name: None)
        self.results = {}

    def _quiet(self, fn, *a, **kw):
        with contextlib.redirect_stdout(io.StringIO()):  # silence RRT chatter
            return fn(*a, **kw)

    def run_goal(self, goal, label):
        """Plan and execute one goal. Returns the number of actions, or None if skipped."""
        plan, self.state = L.solve(self.tplanner, self.statics, self.state, goal, label)
        if plan is None:
            return None
        for act in plan:
            steps, _ = self._quiet(action_steps, self.mplanner, self.grasps, self.q_now, act)
            self.before_action(act)
            self.executor.Execute(steps)
            self.q_now = np.array(steps[-1]["path"][-1])
            self.after_action(act)
        return len(plan)

    def run_all(self, patterns=None, reset=False):
        for name in patterns or L.PATTERN_ORDER:
            if reset and any(self.results.values()):
                self.run_goal(L.start_goal(), "reset before " + name)
            self.results[name] = self.run_goal(L.pattern_goal(L.PATTERNS[name]), name)
            if self.results[name] is not None:
                self.after_pattern(name)
        return self.results

    def go_home(self):
        path = self._quiet(self.mplanner.PlanToConfig, self.q_now, self.q_home)
        self.executor.Execute([{"type": "move", "path": path, "speed": 0.8}])
        self.q_now = self.q_home.copy()

    def print_summary(self, done_text="built"):
        print("\nSummary:")
        for name, n in self.results.items():
            print("   %-8s %s" % (name, "%s (%d actions)" % (done_text, n) if n is not None
                                  else "skipped (infeasible)"))
