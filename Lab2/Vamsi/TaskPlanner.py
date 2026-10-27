import numpy as np

from MotionPlanner import GraspFrame

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0  # clamped to the actuator ctrlrange; squeezes the 4cm block

UP = np.array([0., 0., 1.])


# ---------------------------------------------------------------------- grasp generation
def TopDownGrasps(pos):
    """Top-down TCP poses at pos. The cube is symmetric so all 4 finger directions are valid."""
    axes = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    return [GraspFrame(pos, -UP, f) for f in axes]


def SideGrasps(pos, approach):
    """Horizontal TCP poses approaching along `approach`, fingers closing horizontally."""
    a = np.asarray(approach, dtype=float)
    f = np.cross(a, UP)
    return [GraspFrame(pos, a, f), GraspFrame(pos, a, -f)]


def Offset(T, delta):
    T = np.array(T, dtype=float)
    T[:3, 3] += delta
    return T


# ---------------------------------------------------------------------- task planning
class TaskPlanner:
    """Composes manipulation behaviors into a sequence of executable steps.

    Each step is a dict:
        {"type": "move", "path": [q, ...], "speed": rad/s}    joint-space path to follow
        {"type": "gripper", "value": width}                     open / close the gripper
    """

    def __init__(self, planner, q_home):
        self.planner = planner
        self.q_home = np.array(q_home, dtype=float)
        self.q = self.q_home.copy()
        self.steps = []

    def _Move(self, path, speed):
        self.steps.append({"type": "move", "path": [np.array(q) for q in path], "speed": speed})
        self.q = np.array(path[-1])

    def _Gripper(self, value):
        self.steps.append({"type": "gripper", "value": value})

    def _CartesianChain(self, q_start, waypoints, ignore):
        """Straight-line TCP moves through the given poses. Returns list of joint paths."""
        paths, q = [], q_start
        for T in waypoints:
            path = self.planner.CartesianPath(q, T, ignore=ignore)
            paths.append(path)
            q = path[-1]
        return paths

    def _Interact(self, candidates, pre_offset, post_offsets, gripper_value, ignore, name):
        """Generic grasp/place behavior:
        RRT to a pre-pose, straight-line approach to the target pose, actuate the gripper,
        then straight-line retreat through the post offsets. `ignore` lists obstacles that
        the gripper is allowed to touch (the support surface) during the straight-line moves."""
        for T in candidates:
            T_pre = Offset(T, pre_offset)
            for q_pre in self.planner.IKSolutions(T_pre, [self.q, self.q_home])[:3]:
                try:
                    approach = self._CartesianChain(q_pre, [T], ignore)
                    retreat = self._CartesianChain(approach[-1][-1], [Offset(T, d) for d in post_offsets], ignore)
                    if self.planner.InCollision(retreat[-1][-1]):
                        continue
                    transit = self.planner.PlanToConfig(self.q, q_pre, attempts=1)
                except RuntimeError as e:
                    print("  [%s] candidate rejected: %s" % (name, e))
                    continue
                print("  [%s] planned" % name)
                self._Move(transit, speed=0.8)
                for p in approach:
                    self._Move(p, speed=0.25)
                self._Gripper(gripper_value)
                for p in retreat:
                    self._Move(p, speed=0.25)
                return
        raise RuntimeError("Could not plan behavior: " + name)

    # ------------------------------------------------------------------ behaviors
    def PickTopDown(self, pos, support, clearance=0.12):
        self._Gripper(GRIPPER_OPEN)
        self._Interact(TopDownGrasps(pos), clearance*UP, [clearance*UP],
                       GRIPPER_CLOSED, [support], "pick top-down")

    def PlaceTopDown(self, pos, support, clearance=0.12, drop=0.005):
        self._Interact(TopDownGrasps(np.asarray(pos) + drop*UP), clearance*UP, [clearance*UP],
                       GRIPPER_OPEN, [support], "place top-down")

    def PickSide(self, pos, approach, support, clearance=0.12, lift=0.015):
        a = np.asarray(approach, dtype=float)
        self._Gripper(GRIPPER_OPEN)
        self._Interact(SideGrasps(pos, a), -clearance*a, [lift*UP, lift*UP - clearance*a],
                       GRIPPER_CLOSED, [support], "pick side")

    def PlaceSide(self, pos, approach, support, clearance=0.12, drop=0.01):
        a = np.asarray(approach, dtype=float)
        self._Interact(SideGrasps(np.asarray(pos) + drop*UP, a), -clearance*a, [-clearance*a],
                       GRIPPER_OPEN, [support], "place side")

    def GoHome(self):
        self._Move(self.planner.PlanToConfig(self.q, self.q_home), speed=0.8)
