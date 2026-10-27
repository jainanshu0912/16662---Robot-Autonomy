import time
import pickle
import argparse
import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET

import RobotUtil as rt
from MotionPlanner import CollisionEnv, MotionPlanner
from TaskPlanner import TaskPlanner, GRIPPER_OPEN

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml" 
MODEL_XML = "franka_emika_panda/panda_torque_table_shelves.xml" 

# Home configuration (7-DoF Panda joint angles in radians)
Q_HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])

# Controller gains (computed-torque, per-joint)
KP = np.array([400, 400, 400, 400, 300, 300, 300], dtype=float)
KD = 2*np.sqrt(KP)
SETTLE_DURATION = 0.5  # hold time after each segment

# ---------------------------------------------------------
# Scene description (shared by the simulator and the collision checker)
EndofTable=0.55+0.135+0.05
BLOCKS=[
    ["TablePlane",[EndofTable-0.275,0.,-0.005],[0.275, 0.504, 0.0051]],
    ["LShelfDistal",[EndofTable-0.09-0.0225, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
    ["LShelfProximal",[EndofTable-0.55-0.0225, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
    ["LShelfBack",[EndofTable-0.55-0.0225-0.09, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
    ["LShelfMid",[EndofTable-0.32, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
    ["LShelfArch",[EndofTable-0.275-0.135+0.0225, 0.504-0.045-0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
    ["LShelfBottom",[EndofTable-0.275-0.135+0.0225, 0.504-0.09-0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
    ["LShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSupp2",[EndofTable-0.32-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, 0.504-0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
    ["RShelfDistal",[EndofTable-0.09-0.0225, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
    ["RShelfProximal",[EndofTable-0.55-0.0225, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
    ["RShelfBack",[EndofTable-0.55-0.0225-0.09, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
    ["RShelfMid",[EndofTable-0.32, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
    ["RShelfArch",[EndofTable-0.275-0.135+0.0225, -0.504+0.045+0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
    ["RShelfBottom",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
    ["RShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
    ["RShelfMiddle",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.2],[0.2525, 0.135/2., 0.005]],
    ["RShelfMiddleSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.2],[0.315, 0.0225, 0.0225]],
    ["RShelfTop",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.4],[0.2525, 0.135/2., 0.005]],
    ["RShelfTopSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
    ["RShelfTopSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
    ["RShelfTopSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
    ["RShelfTopSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.4],[0.315, 0.0225, 0.0225]],
]

# Planning-only obstacles (not added to the MuJoCo model)
PLANNING_ONLY_BLOCKS = [
    ["Floor", [0., 0., -0.08], [1.5, 1.5, 0.05]],
]

BLOCK_HALF = 0.02
BLOCK_START = np.array([EndofTable-0.145, 0.0, BLOCK_HALF])

SHELF_TOP = {  # z of the top surface of each shelf plate
    "LShelfBottom": 0.1375+0.01,
    "RShelfBottom": 0.1375+0.01,
    "RShelfMiddle": 0.1375+0.01+0.2,
    "RShelfTop": 0.1375+0.01+0.4,
}
SHELF_EDGE_Y = 0.504-0.09-0.135  # |y| of the shelf edge facing the robot
LEFT_PLACE_Y = SHELF_EDGE_Y+0.025  # block close to the edge so it can be side-grasped
RIGHT_PLACE_Y = -(SHELF_EDGE_Y+0.035)
PLACE_X_RANGE = (0.25, 0.50)

# Block locations on the shelves for each trial (edit these, or use --sample)
TRIALS = [
    {"left_x": 0.45, "right_shelf": "RShelfBottom", "right_x": 0.45},
    {"left_x": 0.35, "right_shelf": "RShelfMiddle", "right_x": 0.30},
    {"left_x": 0.30, "right_shelf": "RShelfTop", "right_x": 0.40},
]


def SampleTrial():
    return {"left_x": np.random.uniform(*PLACE_X_RANGE),
            "right_shelf": str(np.random.choice(["RShelfBottom", "RShelfMiddle", "RShelfTop"])),
            "right_x": np.random.uniform(*PLACE_X_RANGE)}


def TrialTargets(trial):
    left = np.array([trial["left_x"], LEFT_PLACE_Y, SHELF_TOP["LShelfBottom"]+BLOCK_HALF])
    right = np.array([trial["right_x"], RIGHT_PLACE_Y, SHELF_TOP[trial["right_shelf"]]+BLOCK_HALF])
    return left, right


def PlanTrial(planner, trial):
    """Task plan: table -> left shelf (top-down) -> regrasp (side) -> right shelf -> home."""
    left, right = TrialTargets(trial)
    tp = TaskPlanner(planner, Q_HOME)
    tp.PickTopDown(BLOCK_START, support="TablePlane")
    tp.PlaceTopDown(left, support="LShelfBottom")
    tp.PickSide(left, approach=[0., 1., 0.], support="LShelfBottom")
    tp.PlaceSide(right, approach=[0., -1., 0.], support=trial["right_shelf"])
    tp.GoHome()
    return tp.steps


# ---------------------------------------------------------
class SimExecutor:
    """Executes planned steps in MuJoCo with a computed-torque joint controller."""

    def __init__(self, model, data, v=None):
        self.model, self.data, self.v = model, data, v
        self.arm_idx = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_idx = 7
        self.dt = model.opt.timestep
        self.tau_lim = model.actuator_ctrlrange[:7].copy()
        self.M = np.zeros((model.nv, model.nv))
        self.grip = GRIPPER_OPEN
        self.wall_start, self.sim_start = time.time(), data.time

    def Step(self, q_des, qd_des):
        m, d = self.model, self.data
        if hasattr(d, "qM"):  # MuJoCo < 3.4
            mj.mj_fullM(m, self.M, d.qM)
        else:
            mj.mj_fullM(m, d, self.M)
        q, qd = d.qpos[self.arm_idx], d.qvel[self.arm_idx]
        tau = self.M[:7, :7] @ (KP*(q_des - q) + KD*(qd_des - qd)) + d.qfrc_bias[:7]
        d.ctrl[self.arm_idx] = np.clip(tau, self.tau_lim[:, 0], self.tau_lim[:, 1])
        d.ctrl[self.gripper_idx] = self.grip
        mj.mj_step(m, d)
        if self.v is not None:
            self.v.sync()
            ahead = (d.time - self.sim_start) - (time.time() - self.wall_start)
            if ahead > 0:
                time.sleep(ahead)

    def Hold(self, q, duration):
        for _ in range(int(duration/self.dt)):
            self.Step(q, np.zeros(7))

    def ExecuteMove(self, path, speed):
        path = np.array(path)
        if len(path) < 2:
            return
        seg = np.max(np.abs(np.diff(path, axis=0)), axis=1)
        s_cum = np.concatenate([[0.], np.cumsum(seg)])
        L = s_cum[-1]
        # Min-jerk timing along the path; peak joint speed = 1.875 L / T
        T = max(0.5, 1.875*L/speed)
        t = 0.0
        while t <= T:
            s, ds = rt.interp_min_jerk(np.array([0.]), np.array([L]), t, T)
            s, ds = s[0], ds[0]
            i = min(np.searchsorted(s_cum, s, side="right") - 1, len(seg) - 1)
            dq_ds = (path[i+1] - path[i])/max(seg[i], 1e-9)
            q_des = path[i] + (s - s_cum[i])*dq_ds
            self.Step(q_des, dq_ds*ds)
            t += self.dt
        self.Hold(path[-1], SETTLE_DURATION)

    def Execute(self, steps):
        for st in steps:
            if st["type"] == "move":
                self.ExecuteMove(st["path"], st["speed"])
            else:
                self.grip = st["value"]
                self.Hold(self.data.qpos[self.arm_idx].copy(), 1.0)

    def ResetBlock(self, pos):
        # Block freejoint follows the 7 arm + 2 finger joints
        self.data.qpos[9:12] = pos
        self.data.qpos[12:16] = [1., 0., 0., 0.]
        self.data.qvel[9:15] = 0.
        mj.mj_forward(self.model, self.data)

    def BlockPos(self):
        return self.data.qpos[9:12].copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="sample shelf locations instead of TRIALS")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--headless", action="store_true", help="run without the viewer")
    parser.add_argument("--plan-file", default="plans.pkl", help="where to save the joint-space plans")
    args = parser.parse_args()

    #Set random generator seed for reproducing results (useful for debugging)
    np.random.seed(args.seed)

    #Generate scene and obstacle boxes for collison checking (points and axes)
    modelTree = ET.parse(ROOT_MODEL_XML)
    for i in range(len(BLOCKS)):
        rt.add_free_block_to_model(tree=modelTree, name=BLOCKS[i][0], pos=BLOCKS[i][1], density= 20 , size=BLOCKS[i][2] , rgba=[0.2, 0.2, 0.9, 1],free=False)

    #Add a free block to manipulate
    rt.add_free_block_to_model(tree=modelTree, name="Block", pos=[BLOCK_START[0], BLOCK_START[1], 0.05], density= 20 , size=[BLOCK_HALF]*3 , rgba=[0.0, 0.9, 0.2, 1],free=True)

    modelTree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)

    ###### PLAN ######
    trials = [SampleTrial() for _ in range(1)] if args.sample else TRIALS
    planner = MotionPlanner(CollisionEnv(BLOCKS + PLANNING_ONLY_BLOCKS, margin=0.01), seed=args.seed)

    plans = []
    for k, trial in enumerate(trials):
        print("Planning trial %d: %s" % (k+1, trial))
        t0 = time.time()
        plans.append({"trial": trial, "steps": PlanTrial(planner, trial)})
        print("  planned %d steps in %.1fs" % (len(plans[-1]["steps"]), time.time()-t0))

    with open(args.plan_file, "wb") as f:
        pickle.dump(plans, f)
    print("Saved plans to", args.plan_file)

    ###### EXECUTE PLAN ######

    #Load the model
    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)

    # Initialize arm at home
    data.qpos[:7] = Q_HOME
    data.qpos[7:9] = GRIPPER_OPEN
    data.qvel[:7] = 0.0
    mj.mj_forward(model, data)

    #Launch Viewer
    v = None
    if not args.headless:
        v = viewer.launch_passive(model, data)
        v.cam.distance=3.0 #Shift camera 
        v.cam.azimuth += 90

    try:
        ex = SimExecutor(model, data, v)
        for k, p in enumerate(plans):
            print("Executing trial %d" % (k+1))
            ex.ResetBlock(BLOCK_START)
            ex.Hold(Q_HOME, 1.0)
            ex.Execute(p["steps"])
            target = TrialTargets(p["trial"])[1]
            err = np.linalg.norm(ex.BlockPos() - target)
            print("  block at %s, target %s, error %.3fm" % (np.round(ex.BlockPos(), 3), np.round(target, 3), err))
        if v is not None:
            ex.Hold(Q_HOME, 3.0)

    finally:
        if v is not None:
            v.close() #cleanup viewer at the end
