import argparse
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco as mj

import RobotUtil as rt
import BlockGridDomain as dom

# ============================================================================
# Configuration (measure these on the real setup, robot base frame, metres)
# ============================================================================
ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
MODEL_XML = "franka_emika_panda/panda_lab3_scene.xml"

BLOCK_HALF = 0.02             # 4 cm cubes
GRID_CENTER = np.array([0.50, 0.0])
GRID_SPACING = 0.06           # centre-to-centre: 2 cm gap, so an open finger hits a neighbour
HOME_X = [0.33, 0.42, 0.51, 0.60]   # 9 cm apart so open fingers never touch a neighbour;
                                    # x <= 0.60 keeps the arm clear of frankapy's front wall
RED_HOME_Y = 0.15             # robot's left (+y). Not further out: the 20 cm wide hand would
BLUE_HOME_Y = -0.15           # hit the bottom side shelves during y-axis grasps
TABLE_Z = 0.0                 # height of the table surface

# ---- Optional calibration from the real setup (written by calibrate.py) ----
# If calibration.json exists, block locations and table height come from measured points.
CALIBRATION_FILE = "calibration.json"
CALIB = None
if os.path.exists(CALIBRATION_FILE):
    import json
    with open(CALIBRATION_FILE) as _f:
        CALIB = {k: np.array(v, dtype=float) for k, v in json.load(_f).items()}
    # measured points are TCP positions at grasp height (block centre)
    TABLE_Z = float(np.mean([p[2] for p in CALIB.values()])) - BLOCK_HALF
    print(f"Using {CALIBRATION_FILE}: {len(CALIB)} measured points, table z = {TABLE_Z:.4f}")

TCP_OFFSET = 0.1034           # hand frame -> centre of the finger pads (Franka hand)
GRASP_Z = TABLE_Z + BLOCK_HALF            # TCP height when grasping a block on the table
PLACE_Z = TABLE_Z + BLOCK_HALF + 0.003    # release a few mm above the table
APPROACH_Z = TABLE_Z + 0.12               # pre-grasp / pre-place height
TRANSIT_Z = TABLE_Z + 0.16                # height for moving between locations

# frankapy virtual safety walls (approximate inner faces, robot base frame) and the
# half-length of the Franka hand along its finger axis. A grasp axis is disabled at a
# location if the hand would reach past a wall (with margin).
WALL_FRONT_X = 0.75
WALL_SIDE_Y = 0.46
HAND_HALF_LENGTH = 0.105
WALL_MARGIN = 0.07   # generous: frankapy pads its link boxes, and x = 0.66 already failed its check

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0125

HOME_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])   # Lab 2 start configuration

# Timing (seconds) - slow these down for the real robot if needed
T_LONG = 2.5        # transit segments
T_SHORT = 1.5       # vertical approach / retreat
T_GRIPPER = 0.8     # time given to the gripper to open/close

# Simulation-only joint PD gains (4x the Lab 2 stiffness so the arm actually reaches each
# grasp pose before the gripper acts). The real robot uses its own controller.
KP = np.array([480, 480, 400, 360, 240, 160, 120], dtype=float)
KD = np.array([16, 16, 12, 10, 8, 6, 4], dtype=float)
SETTLE_TOL = 0.002     # rad: arm must be this close to the target before the gripper moves
SETTLE_MAX = 1.5       # s

ARM_IDX = list(range(7))
GRIPPER_IDX = 7


# ============================================================================
# Geometry: symbolic location -> table position
# ============================================================================
def location_xy(loc):
    if CALIB is not None:
        return calibrated_xy(loc)
    if loc.startswith("g"):
        r, c = int(loc[1]), int(loc[2])
        # row 0 = far from the robot (+x), column 0 = robot's left (+y)
        return GRID_CENTER + np.array([(1 - r) * GRID_SPACING, (1 - c) * GRID_SPACING])
    i = int(loc[2])
    if loc.startswith("hr"):
        return np.array([HOME_X[i], RED_HOME_Y])
    if loc.startswith("hb"):
        return np.array([HOME_X[i], BLUE_HOME_Y])
    raise ValueError(f"Unknown location {loc}")


def calibrated_xy(loc):
    """Location from measured reference points: an exact measurement if there is one,
    otherwise bilinear interpolation between the 4 measured grid corners, or linear
    interpolation between the first and last measured home slot of that colour."""
    if loc in CALIB:
        return CALIB[loc][:2].copy()
    if loc.startswith("g"):
        r, c = int(loc[1]) / 2.0, int(loc[2]) / 2.0
        far = (1 - c) * CALIB["g00"] + c * CALIB["g02"]
        near = (1 - c) * CALIB["g20"] + c * CALIB["g22"]
        return ((1 - r) * far + r * near)[:2]
    side = loc[:2]                      # "hr" or "hb"
    s = int(loc[2]) / 3.0
    return ((1 - s) * CALIB[f"{side}0"] + s * CALIB[f"{side}3"])[:2]


def wall_blocked_axes():
    """Grasp axes that would push the hand into frankapy's virtual walls, per location."""
    blocked = {}
    for loc in dom.GRID_CELLS + dom.RED_HOME + dom.BLUE_HOME:
        x, y = location_xy(loc)
        axes = set()
        if x + HAND_HALF_LENGTH > WALL_FRONT_X - WALL_MARGIN:      # fingers along x
            axes.add("x")
        if abs(y) + HAND_HALF_LENGTH > WALL_SIDE_Y - WALL_MARGIN:  # fingers along y
            axes.add("y")
        if axes:
            blocked[loc] = axes
    return blocked


# ============================================================================
# Scene
# ============================================================================
END_OF_TABLE = 0.55 + 0.135 + 0.05
STATIC_BOXES = [  # same table and shelves as Lab 2 (name, centre, half-sizes)
    ["TablePlane", [END_OF_TABLE - 0.275, 0., -0.005], [0.275, 0.504, 0.0051]],
    ["LShelfDistal", [END_OF_TABLE - 0.09 - 0.0225, 0.504 - 0.045 - 0.0225, 0.315], [0.0225, 0.0225, 0.315]],
    ["LShelfProximal", [END_OF_TABLE - 0.55 - 0.0225, 0.504 - 0.045 - 0.0225, 0.3825 - 0.135], [0.0225, 0.0225, 0.3825]],
    ["LShelfBack", [END_OF_TABLE - 0.55 - 0.0225 - 0.09, 0.504 - 0.045 - 0.0225, 0.3825 - 0.135], [0.0225, 0.0225, 0.3825]],
    ["LShelfMid", [END_OF_TABLE - 0.32, 0.504 - 0.045 - 0.0225, 0.315], [0.0225, 0.0225, 0.315]],
    ["LShelfArch", [END_OF_TABLE - 0.275 - 0.135 + 0.0225, 0.504 - 0.045 - 0.0225, 0.63 + 0.0225], [0.315, 0.0225, 0.0225]],
    ["LShelfBottom", [END_OF_TABLE - 0.275 - 0.135 + 0.0225, 0.504 - 0.09 - 0.135 / 2., 0.1375 + 0.005], [0.2525, 0.135 / 2., 0.005]],
    ["RShelfDistal", [END_OF_TABLE - 0.09 - 0.0225, -0.504 + 0.045 + 0.0225, 0.315], [0.0225, 0.0225, 0.315]],
    ["RShelfProximal", [END_OF_TABLE - 0.55 - 0.0225, -0.504 + 0.045 + 0.0225, 0.3825 - 0.135], [0.0225, 0.0225, 0.3825]],
    ["RShelfBack", [END_OF_TABLE - 0.55 - 0.0225 - 0.09, -0.504 + 0.045 + 0.0225, 0.3825 - 0.135], [0.0225, 0.0225, 0.3825]],
    ["RShelfMid", [END_OF_TABLE - 0.32, -0.504 + 0.045 + 0.0225, 0.315], [0.0225, 0.0225, 0.315]],
    ["RShelfArch", [END_OF_TABLE - 0.275 - 0.135 + 0.0225, -0.504 + 0.045 + 0.0225, 0.63 + 0.0225], [0.315, 0.0225, 0.0225]],
    ["RShelfBottom", [END_OF_TABLE - 0.275 - 0.135 + 0.0225, -0.504 + 0.09 + 0.135 / 2., 0.1375 + 0.005], [0.2525, 0.135 / 2., 0.005]],
    ["RShelfMiddle", [END_OF_TABLE - 0.275 - 0.135 + 0.0225, -0.504 + 0.09 + 0.135 / 2., 0.1375 + 0.005 + .2], [0.2525, 0.135 / 2., 0.005]],
    ["RShelfTop", [END_OF_TABLE - 0.275 - 0.135 + 0.0225, -0.504 + 0.09 + 0.135 / 2., 0.1375 + 0.005 + .4], [0.2525, 0.135 / 2., 0.005]],
]
RED_RGBA = [0.9, 0.1, 0.1, 1]
BLUE_RGBA = [0.0, 0.45, 0.75, 1]


def add_box(tree, name, pos, size, rgba, free=False, density=20, collide=True):
    worldbody = tree.getroot().find("worldbody")
    body = ET.SubElement(worldbody, "body", {"name": name, "pos": f"{pos[0]} {pos[1]} {pos[2]}"})
    attrs = {"type": "box", "density": f"{density}",
             "size": f"{size[0]} {size[1]} {size[2]}",
             "rgba": " ".join(str(v) for v in rgba)}
    if free:
        attrs.update({"friction": "1.5 0.01 0.001", "condim": "4"})
    if not collide:
        attrs.update({"contype": "0", "conaffinity": "0"})
    ET.SubElement(body, "geom", attrs)
    if free:
        ET.SubElement(body, "freejoint")


def build_scene(placement):
    tree = ET.parse(ROOT_MODEL_XML)
    # keep the ground plane below the table, even if the measured table is below z = 0
    for g in tree.getroot().find("worldbody").findall("geom"):
        if g.get("name") == "ground":
            g.set("pos", f"0 0 {min(0.0, TABLE_Z) - 0.02}")
    for name, pos, size in STATIC_BOXES:
        if name == "TablePlane":   # match the (measured) table height
            pos = [pos[0], pos[1], pos[2] + TABLE_Z]
        add_box(tree, name, pos, size, [0.6, 0.6, 0.65, 1])
    # grey grid squares (visual only)
    for l in dom.GRID_CELLS:
        x, y = location_xy(l)
        add_box(tree, f"cell_{l}", [x, y, TABLE_Z + 0.0006], [0.024, 0.024, 0.0005],
                [0.75, 0.75, 0.75, 1], collide=False)
    for b, l in placement.items():
        x, y = location_xy(l)
        rgba = RED_RGBA if dom.block_color(b) == "red" else BLUE_RGBA
        add_box(tree, f"block_{b}", [x, y, TABLE_Z + BLOCK_HALF + 0.001],
                [BLOCK_HALF] * 3, rgba, free=True)
    tree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)
    return mj.MjModel.from_xml_path(MODEL_XML)


# ============================================================================
# Inverse kinematics (damped least squares on the MuJoCo model)
# ============================================================================
def topdown_rotation(yaw):
    """Gripper pointing straight down; fingers close along (cos yaw, sin yaw, 0)."""
    y_h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z_h = np.array([0.0, 0.0, -1.0])
    x_h = np.cross(y_h, z_h)
    return np.column_stack([x_h, y_h, z_h])


class IKSolver:
    def __init__(self, model):
        self.model = model
        self.data = mj.MjData(model)
        self.hand_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")
        self.lo = model.jnt_range[:7, 0].copy() + 0.02
        self.hi = model.jnt_range[:7, 1].copy() - 0.02

    def fk(self, q):
        self.data.qpos[:7] = q
        mj.mj_kinematics(self.model, self.data)
        R = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        p = self.data.xpos[self.hand_id] + R @ np.array([0, 0, TCP_OFFSET])
        return p, R

    def solve(self, p_des, R_des, q_seed, iters=300, tol_p=5e-4, tol_r=5e-3):
        q = q_seed.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        lam = 0.05
        for _ in range(iters):
            p, R = self.fk(q)
            e_p = p_des - p
            axis, ang = rt.R2axisang(np.clip(R_des @ R.T, -1, 1))
            e_r = np.array(axis) * ang
            if np.linalg.norm(e_p) < tol_p and np.linalg.norm(e_r) < tol_r:
                return q, True
            mj.mj_comPos(self.model, self.data)
            mj.mj_jac(self.model, self.data, jacp, jacr, p, self.hand_id)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            # small pull towards the seed keeps the arm in the same posture family
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.05 * (q_seed - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q, False


# ============================================================================
# Move -> joint waypoints
# ============================================================================
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class CollisionChecker:
    """Checks a joint configuration for contact between the robot and the static scene
    (table + shelves) using the MuJoCo model. Blocks are ignored."""

    def __init__(self, model):
        self.model = model
        self.data = mj.MjData(model)
        static = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n, _, _ in STATIC_BOXES}
        static.add(0)  # world body (ground plane)
        robot_names = ["link1", "link2", "link3", "link4", "link5", "link6", "link7",
                       "hand", "left_finger", "right_finger"]
        robot = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in robot_names}
        self.pairs = (static, robot)

    def collides(self, q, grip=GRIPPER_OPEN):
        self.data.qpos[:7] = q
        self.data.qpos[7:9] = grip
        mj.mj_forward(self.model, self.data)
        static, robot = self.pairs
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 in static and b2 in robot) or (b2 in static and b1 in robot):
                if c.dist < -0.001:
                    return (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, b1),
                            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, b2))
        return None


class WaypointBuilder:
    def __init__(self, model):
        self.ik = IKSolver(model)
        self.cc = CollisionChecker(model)
        self.q = HOME_Q.copy()
        self.yaw = None
        self.waypoints = []

    def _add(self, q, grip, duration):
        hit = self.cc.collides(q, grip)
        if hit is not None:
            raise RuntimeError(f"Waypoint collides with the scene: {hit[0]} <-> {hit[1]}")
        self.waypoints.append((q.copy(), grip, duration))
        self.q = q.copy()

    def _solve_topdown(self, xy, axis, z_low):
        """Finds a matching pair of top-grasp configurations above (TRANSIT_Z) and at (z_low)
        a location, with the fingers closing along `axis`. Rotating the gripper by 180 deg gives
        the same grasp, so both yaws are tried (seeded from the current posture and from
        HOME_Q). Pairs that touch the table/shelves are rejected; the pair needing the least
        joint motion from the current posture is returned."""
        base = 0.0 if axis == "x" else np.pi / 2
        best = None
        for yaw in (base, wrap(base + np.pi)):
            R = topdown_rotation(yaw)
            for seed in (self.q, HOME_Q):
                q_top, ok = self.ik.solve(np.array([xy[0], xy[1], TRANSIT_Z]), R, seed)
                if not ok:
                    continue
                q_low, ok = self.ik.solve(np.array([xy[0], xy[1], z_low]), R, q_top)
                if not ok or np.max(np.abs(q_low - q_top)) > 1.0:
                    continue
                if self.cc.collides(q_top) or self.cc.collides(q_low):
                    continue
                cost = np.max(np.abs(q_top - self.q))
                if best is None or cost < best[0]:
                    best = (cost, q_top, q_low, yaw)
        if best is None:
            raise RuntimeError(f"No reachable collision-free top grasp at {xy}, fingers along {axis}")
        return best[1], best[2], best[3]

    def _vertical(self, xy, q_end, yaw, grip, duration):
        """Straight up/down move between two verified configurations, with one IK midpoint
        so the fingers stay on the vertical line."""
        z0 = self.ik.fk(self.q)[0][2]
        z1 = self.ik.fk(q_end)[0][2]
        q_mid, ok = self.ik.solve(np.array([xy[0], xy[1], 0.5 * (z0 + z1)]),
                                  topdown_rotation(yaw), 0.5 * (self.q + q_end))
        if ok and np.max(np.abs(q_mid - 0.5 * (self.q + q_end))) < 0.3:
            self._add(q_mid, grip, duration / 2)
            self._add(q_end, grip, duration / 2)
        else:
            self._add(q_end, grip, duration)

    def _transit(self, q_goal, grip, duration, depth=0):
        """Joint-space move between two poses at transit height. If the straight joint path
        dips too low above the table, insert a raised midpoint and split the move."""
        n = 25
        zs = [self.ik.fk(self.q + (q_goal - self.q) * k / n)[0][2] for k in range(n + 1)]
        if min(zs) < APPROACH_Z and depth < 3:
            q_mid = 0.5 * (self.q + q_goal)
            p_mid, R_mid = self.ik.fk(q_mid)
            p_mid[2] = TRANSIT_Z + 0.05
            q_mid, ok = self.ik.solve(p_mid, R_mid, q_mid)
            if ok:
                self._transit(q_mid, grip, duration / 2, depth + 1)
                self._transit(q_goal, grip, duration / 2, depth + 1)
                return
        self._add(q_goal, grip, duration)

    def pick_and_place(self, src, dst, pick_axis, place_axis):
        src_xy, dst_xy = location_xy(src), location_xy(dst)
        # pick
        q_top, q_low, yaw = self._solve_topdown(src_xy, pick_axis, GRASP_Z)
        self._transit(q_top, GRIPPER_OPEN, T_LONG)
        self._vertical(src_xy, q_low, yaw, GRIPPER_OPEN, T_SHORT)
        self._add(self.q, GRIPPER_CLOSED, T_GRIPPER)                 # close
        self._vertical(src_xy, q_top, yaw, GRIPPER_CLOSED, T_SHORT)
        # place
        q_top, q_low, yaw = self._solve_topdown(dst_xy, place_axis, PLACE_Z)
        self._transit(q_top, GRIPPER_CLOSED, T_LONG)
        self._vertical(dst_xy, q_low, yaw, GRIPPER_CLOSED, T_SHORT)
        self._add(self.q, GRIPPER_OPEN, T_GRIPPER)                   # open
        self._vertical(dst_xy, q_top, yaw, GRIPPER_OPEN, T_SHORT)

    def go_home(self):
        self._transit(HOME_Q, GRIPPER_OPEN, T_LONG)

    def as_array(self):
        return np.array([np.concatenate([q, [g]]) for q, g, _ in self.waypoints])

    def durations(self):
        return np.array([d for _, _, d in self.waypoints])


# ============================================================================
# Simulation executor (Lab 2 PD + gravity compensation, min-jerk)
# ============================================================================
class SimExecutor:
    def __init__(self, model, viewer=None):
        self.model = model
        self.data = mj.MjData(model)
        self.data.qpos[ARM_IDX] = HOME_Q
        self.data.qpos[7:9] = GRIPPER_OPEN
        mj.mj_forward(model, self.data)
        self.q_cmd = HOME_Q.copy()
        self.grip = GRIPPER_OPEN
        self.viewer = viewer
        self.dt = model.opt.timestep
        self.recorder = None   # optional callable(data), called every step

    def _step(self, q_des, qd_des):
        q = self.data.qpos[ARM_IDX]
        qd = self.data.qvel[ARM_IDX]
        tau = KP * (q_des - q) + KD * (qd_des - qd)
        self.data.ctrl[ARM_IDX] = tau + self.data.qfrc_bias[:7]
        self.data.ctrl[GRIPPER_IDX] = self.grip
        mj.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        if self.recorder is not None:
            self.recorder(self.data)

    def _settle(self):
        """Hold the last commanded pose until the arm has actually reached it."""
        for _ in range(int(SETTLE_MAX / self.dt)):
            if np.max(np.abs(self.data.qpos[ARM_IDX] - self.q_cmd)) < SETTLE_TOL:
                break
            self._step(self.q_cmd, np.zeros(7))

    def run(self, waypoints, durations):
        for wp, T in zip(waypoints, durations):
            q_goal, grip = wp[:7], wp[7]
            if grip != self.grip:                      # gripper action: hold arm still
                self._settle()
                self.grip = grip
                for _ in range(int(T / self.dt)):
                    self._step(self.q_cmd, np.zeros(7))
                continue
            q_start = self.q_cmd.copy()
            n = max(1, int(T / self.dt))
            for k in range(n):
                q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, k * self.dt, T)
                self._step(q_des, qd_des)
            self.q_cmd = q_goal.copy()
            self._settle()      # don't start the next segment while the arm is still swinging
        for _ in range(int(0.3 / self.dt)):            # settle
            self._step(self.q_cmd, np.zeros(7))

    def block_xy(self, block):
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"block_{block}")
        return self.data.xpos[bid].copy()


# ============================================================================
# Main
# ============================================================================
PATTERN_ORDER = ["pickaxe", "axe", "box", "bow"]


def print_grid(placement):
    occ = {l: dom.block_color(b)[0].upper() for b, l in placement.items()}
    for r in range(3):
        print("      " + " ".join(occ.get(dom.cell(r, c), ".") for c in range(3)))


def main():
    np.random.seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="simulate without the viewer")
    ap.add_argument("--plan-only", action="store_true", help="only print symbolic plans")
    ap.add_argument("--reset", action="store_true", help="return blocks home between patterns")
    ap.add_argument("--out", default="waypoints", help="folder for the saved joint trajectories")
    ap.add_argument("--record", metavar="FILE.mp4", help="save a video of the simulation (needs ffmpeg)")
    args = ap.parse_args()

    dom.AXIS_BLOCKED = wall_blocked_axes()
    for loc, axes in sorted(dom.AXIS_BLOCKED.items()):
        print(f"{loc}: fingers along {'/'.join(sorted(axes))} not allowed (hand would reach a virtual wall)")
    planner = dom.FastPlanner(dom.make_Grid_domain())
    placement = dom.make_initial_state()
    model = build_scene(placement)
    builder = WaypointBuilder(model)

    # ---------- 1. Task planning ----------
    stages = []   # (name, [(src, dst, block, pick_axis, place_axis), ...])
    for name in PATTERN_ORDER:
        goals = [(name, dom.goal_from_pattern(dom.PATTERNS[name]))]
        if args.reset and stages and dom.pattern_is_buildable(dom.PATTERNS[name]):
            goals.insert(0, (f"reset before {name}", dom.home_goal()))
        for label, goal in goals:
            if label == name and not dom.pattern_is_buildable(dom.PATTERNS[name]):
                print(f"\n[{label}] infeasible: no placement order avoids finger collisions -> skipping")
                continue
            t0 = time.time()
            plan = planner.plan(dom.state_from_placement(placement), goal)
            if plan is None:
                print(f"\n[{label}] infeasible (no plan found) -> skipping")
                continue
            print(f"\n[{label}] {len(plan)} moves, planned in {time.time() - t0:.1f}s")
            moves = []
            for i, a in enumerate(plan, 1):
                src, dst, color, pa, qa = dom.parse_move(a)
                block = dom.apply_move_to_placement(placement, src, dst)
                moves.append((src, dst, block, pa, qa))
                print(f"  {i:02d}. move {block} ({color}) {src} -> {dst}   "
                      f"pick fingers along {pa}, place fingers along {qa}")
            print("    result:")
            print_grid(placement)
            stages.append((label, moves))

    if args.plan_only:
        return

    # ---------- 2. Motion: joint waypoints for every stage ----------
    os.makedirs(args.out, exist_ok=True)
    stage_wps = []
    for label, moves in stages:
        builder.waypoints = []
        labels = []
        for i, (src, dst, block, pa, qa) in enumerate(moves, 1):
            n0 = len(builder.waypoints)
            builder.pick_and_place(src, dst, pa, qa)
            labels += [f"move {i}: {block} {src}->{dst} (pick along {pa}, place along {qa})"] * (len(builder.waypoints) - n0)
        n0 = len(builder.waypoints)
        builder.go_home()
        labels += ["return home"] * (len(builder.waypoints) - n0)
        W, D = builder.as_array(), builder.durations()
        tcp = np.array([builder.ik.fk(w[:7])[0] for w in W])
        fname = os.path.join(args.out, label.replace(" ", "_") + ".npz")
        np.savez(fname, waypoints=W, durations=D, labels=np.array(labels), tcp=tcp)
        print(f"Saved {len(W)} waypoints for '{label}' to {fname}")
        stage_wps.append((label, moves, W, D))

    # ---------- 3. Execute in simulation ----------
    sim = SimExecutor(model)
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjv
        viewer = mjv.launch_passive(model, sim.data)
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -50
        viewer.cam.lookat[:] = [0.5, 0.0, 0.1]
        sim.viewer = viewer

    writer = None
    if args.record:
        import subprocess
        W_PX, H_PX, FPS = 640, 480, 30   # 640x480 is the default offscreen buffer size
        renderer = mj.Renderer(model, H_PX, W_PX)
        cam = mj.MjvCamera()
        cam.lookat[:] = [0.5, 0.0, 0.05]
        cam.distance, cam.azimuth, cam.elevation = 1.1, 180, -55
        writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W_PX}x{H_PX}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                   "-preset", "veryfast", "-pix_fmt", "yuv420p", args.record], stdin=subprocess.PIPE)
        every = max(1, int(round(1.0 / (FPS * sim.dt))))
        counter = {"k": 0}

        def record(data):
            counter["k"] += 1
            if counter["k"] % every == 0:
                renderer.update_scene(data, cam)
                writer.stdin.write(renderer.render().tobytes())
        sim.recorder = record

    try:
        expected = dom.make_initial_state()
        for label, moves, W, D in stage_wps:
            print(f"\nExecuting '{label}' in simulation ...")
            sim.run(W, D)
            for src, dst, block, pa, qa in moves:
                expected[block] = dst
            errs = []
            for block, loc in expected.items():
                p = sim.block_xy(block)
                errs.append((block, loc, np.linalg.norm(p[:2] - location_xy(loc)), p[2]))
            worst = max(errs, key=lambda e: e[2])
            ok = all(e[2] < 0.015 and abs(e[3] - BLOCK_HALF) < 0.01 for e in errs)
            print(f"  blocks where the plan expects them: {'YES' if ok else 'NO'} "
                  f"(worst xy error {worst[2] * 1000:.1f} mm, {worst[0]} at {worst[1]})")
    finally:
        if viewer is not None:
            viewer.close()
        if writer is not None:
            writer.stdin.close()
            writer.wait()
            print(f"Saved simulation video to {args.record}")


if __name__ == "__main__":
    main()
