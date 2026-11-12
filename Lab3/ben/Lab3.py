import itertools
import time
from typing import Iterable, Set, List

import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
import RobotUtil as rt
from lab3.SymPlanner import ActionSchema, Predicate, Planner

# TO UNCOMMENT
# from frankapy import FrankaArm


ArmPos = np.ndarray  # 7 (NUM_JOINTS) elements
HomogenousPose = np.ndarray  # 4x4, where [:3, :3] is RotationMatrix, [3, :] is [0, 0, 0, 1]
RotationMatrix = np.ndarray  # 3x3, orthogonal
Pos3D = np.ndarray  # 3 elements
PoseXYZRPY = np.ndarray # 6 elements


class Pose:
    def __init__(self, H: HomogenousPose):
        assert H.shape == (4, 4)
        assert np.allclose(H[3, :], [0., 0., 0., 1.])
        # im just gonna assume R is good
        self._H = H

    @property
    def H(self) -> HomogenousPose:
        return self._H.copy()

    @property
    def R(self) -> RotationMatrix:
        return self.H[:3, :3]

    @property
    def t(self) -> Pos3D:
        return self.H[:3, 3]

    @property
    def rpy(self) -> Pos3D:
        return Rotation.from_matrix(self.R).as_euler("xyz")

    @staticmethod
    def from_H(H: HomogenousPose) -> 'Pose':
        return Pose(H)

    @staticmethod
    def from_Rt(R: RotationMatrix, t: Pos3D) -> 'Pose':
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t
        return Pose(H)

    @staticmethod
    def from_xyz_rpy(xyzrpy: PoseXYZRPY) -> 'Pose':
        return Pose(rt.rpyxyz2H(xyzrpy[3:], xyzrpy[:3]))

    @staticmethod
    def from_data_body(data: mj.MjData, body_idx: int) -> 'Pose':
        R = data.xmat[body_idx].reshape((3, 3))
        t = data.xpos[body_idx]
        return Pose.from_Rt(R, t)

    def __eq__(self, other: 'Pose') -> bool:
        return np.allclose(self, other)

    def __repr__(self) -> str:
        xyz = self.t
        rpy = np.rad2deg(self.rpy)
        return f"Pose(xyz={np.round(xyz, 3)}, rpy={np.round(rpy, 1)}"

    def compute_error(self, other: 'Pose') -> PoseXYZRPY:
        curr_err = np.zeros((6,))
        curr_err[:3] = other.t - self.t
        rot_err = other.R @ self.R.T
        err_axis, theta = rt.R2axisang(rot_err)
        curr_err[3:] = np.array(err_axis) * theta
        return curr_err

    def compute_norm_error(self, other: 'Pose') -> float:
        return float(np.linalg.norm(self.compute_error(other)))


@dataclass
class Block:
    name: str
    pos: Pos3D
    size: Pos3D
    density: float = 20.
    rgba: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.9, 1]) # 4D, each 0-1
    free: bool = False

    def add_to_model(self, model_tree):
        rt.add_free_block_to_model(model_tree, name=self.name, pos=self.pos, size=self.size, density=self.density, rgba=self.rgba, free=self.free)


class Arm:
    def __init__(
            self,
            model: mj.MjModel,
            data: mj.MjData,
            view: mj.viewer.Handle | None,
            is_simulated: bool,
            initial_pose: ArmPos | None = None,
            num_joints: int = 7,
            gripper_open: float = 0.04,
            gripper_closed: float = 0.0125,
            kp: ArmPos | None = None,
            kd: ArmPos | None = None,
            body_idxs_for_collision_to_ignore: Set[int] | None = None
    ):
        """
        Initialize a new Franka Arm

        Args:
            model: the mujuco model containing the data of the environment
            data: the mujuco data object holding the changing data of the environment
            view: the mujoco viewer (if simulated) - ignored if not simulated.
            is_simulated: whether this robot is simulated or running on the actual arm
            initial_pose: the initial pose of the robot
            num_joints: the number of joints on the arm of the robot (not including gripper)
            gripper_open: the position of the gripper when it is open
            gripper_closed: the position of the gripper when it is closed
            kp: the P coefficients for PID (ignored if not simulated)
            kd: the D coefficients for PID (ignored if not simulated)
            body_idxs_for_collision_to_ignore: body indexes to ignore when testing for collision
        """
        self._model = model
        self._data = data
        self._view = view
        self._test_data = mj.MjData(self._model)

        self._arm_idxs = list(range(num_joints))
        self._gripper_idx = num_joints
        self._data.qpos[self._arm_idxs] = np.zeros((num_joints,)) if initial_pose is None else initial_pose
        self._data.qvel[self._arm_idxs] = 0.
        self._kp = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float) if kp is None else kp
        self._kd = np.array([8, 8, 6, 5, 4, 3, 2], dtype=float) if kd is None else kd
        assert self._kp.shape == (num_joints,), f"{kp.shape}, {num_joints}, {kp}"
        assert self._kd.shape == (num_joints,)

        self._gripper_body_id = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_BODY, "hand")
        self._gripper_open = gripper_open
        self._gripper_closed = gripper_closed

        self._time = 0.
        self._is_simulated = is_simulated
        self._body_idxs_to_ignore = body_idxs_for_collision_to_ignore

        if self.is_simulated:
            self._franka = None
        # TO UNCOMMENT
        # else:
        #     self._franka = FrankaArm()
        #     self._franka.reset_joints()

        self._update()

    @property
    def is_simulated(self) -> bool:
        """
        Returns:
            whether this arm is simulated
        """
        return self._is_simulated

    @property
    def time(self) -> float:
        """
        NOTE: THIS IS ONLY ACCURATE IN SIMULATION

        Returns:
            the current elapsed time
        """
        return self._time

    @property
    def _dt(self) -> float:
        """
        Returns:
            the timestep of the model
        """
        return self._model.opt.timestep

    @property
    def gripper_open(self) -> float:
        """
        Returns:
            the position of the gripper when it is open
        """
        return self._gripper_open

    @property
    def gripper_closed(self) -> float:
        """
        Returns:
            the position of the gripper when it is closed
        """
        return self._gripper_closed

    @property
    def q(self) -> ArmPos:
        """
        Returns:
            the current position of the arm joints
        """
        return self._data.qpos[self._arm_idxs].copy()

    @property
    def q_vel(self) -> ArmPos:
        """
        Returns:
            the current velocity of the arm joints
        """
        return self._data.qvel[self._arm_idxs].copy()

    @property
    def lo(self) -> ArmPos:
        """
        Returns:
            the minimum position each arm joint can move to
        """
        return self._model.jnt_range[self._arm_idxs, 0]

    @property
    def hi(self) -> ArmPos:
        """
        Returns:
            the maximum position each arm joint can move to
        """
        return self._model.jnt_range[self._arm_idxs, 1]

    @property
    def gripper_pose(self) -> Pose:
        """
        Returns:
            the current Pose of the gripper
        """
        return Pose.from_data_body(self._data, self._gripper_body_id)

    def sample_pos(self) -> ArmPos:
        """
        Returns:
            a random arm position, uniformly sampled from within its joint range
        """
        return np.random.uniform(self.lo, self.hi)

    def sample_gripper_pos(self) -> float:
        """
        Returns:
            a random gripper position, randomly sampled between open and closed
        """
        return np.random.uniform(self.gripper_closed, self.gripper_open)

    def _set_test_q(self, idxs: list[int], q: np.ndarray | float):
        self._test_data.qpos[idxs] = q
        mj.mj_forward(self._model, self._test_data)

    def ik(self, target_pose: Pose, init_arm_q: ArmPos | None = None, max_iter: int = 2000, x_eps=1e-3, r_eps=1e-3) -> ArmPos | None:
        """
        Determine the joint angles needed to achieve a target gripper pose

        Args:
            target_pose: the target pose for the gripper to reach
            init_arm_q: the starting position of the arm joints, or None for a random sample
            max_iter: the maximum number of iterations to run
            x_eps: the acceptable range for the translation error (meters)
            r_eps: the acceptable range for the rotation error (radians)

        Returns:
            an ArmPos which achieves the given target_pose, or None if no position could be found
        """
        W = np.diag([1., 1., 100., 100., 1., 1., 100.])
        C = np.diag([1000000., 1000000., 1000000., 1000., 1000., 1000.])
        W_inv = np.linalg.inv(W)
        C_inv = np.linalg.inv(C)
        step_size = 0.01
        current = self.q if init_arm_q is None else init_arm_q
        for iteration in range(max_iter):
            self._set_test_q(self._arm_idxs, current)
            hand_pose = Pose.from_data_body(self._test_data, self._gripper_body_id)

            jac_pos = np.zeros((3, self._model.nv))
            jac_rot = np.zeros((3, self._model.nv))
            mj.mj_jacBody(self._model, self._test_data, jac_pos, jac_rot, self._gripper_body_id)
            J = np.vstack((jac_pos, jac_rot))[:, self._arm_idxs]

            raw_err = hand_pose.compute_error(target_pose)
            curr_err = raw_err
            err_norm = np.linalg.norm(curr_err)
            if err_norm > step_size:
                curr_err = (curr_err / err_norm) * step_size

            t_dist = np.linalg.norm(curr_err[:3])
            r_dist = np.linalg.norm(curr_err[3:])
            if t_dist < x_eps and r_dist < r_eps:
                break

            pseudo_J = W_inv @ J.T @ np.linalg.inv(J @ W_inv @ J.T + C_inv)
            delta_q = pseudo_J @ curr_err
            current += delta_q
            current = np.clip(current, self.lo, self.hi)
        if np.linalg.norm(raw_err) > step_size:
            return None
        return current

    def ik_multi(self, target_pose: Pose, force_guess: ArmPos | None = None, num_tries: int = 20, **ik_kwargs) -> ArmPos | None:
        """
        Run several IK calls with the hope that one converges, returning the first valid joint position which achieves
        the target_pose

        Args:
            target_pose: the target pose for the gripper to reach
            force_guess: an initial arm position which one iteration of the IK solver will use when solving
            num_tries: the number of times to run IK
            **ik_kwargs: other arguments for IK - see self.ik

        Returns:
            an ArmPos which achieves the given target_pose, or None if no position could be found
        """
        for idx in range(num_tries):
            start_pos = force_guess if force_guess is not None and idx == 0 else self.sample_pos()
            soln = self.ik(target_pose, start_pos, **ik_kwargs)
            if soln is None:
                print("Could not reach")
            elif self.is_in_collision(soln):
                print("Solution is in collision")
            if soln is not None and not self.is_in_collision(soln):
                return soln
        return None

    def is_in_collision(self, q: ArmPos) -> bool:
        """
        Test if a given arm position is in collision with the environment

        Ignores any collisions with bodies given by body_idxs_for_collision_to_ignore on init

        Args:
            q: the arm position to test for collision

        Returns:
            True if the given q results in a collision, False otherwise
        """
        self._set_test_q(self._arm_idxs, q)
        self._set_test_q([self._gripper_idx], self.gripper_open)
        for contact in self._test_data.contact:
            body1_id = self._model.geom_bodyid[contact.geom1]
            body2_id = self._model.geom_bodyid[contact.geom2]
            if body1_id not in self._body_idxs_to_ignore and body2_id not in self._body_idxs_to_ignore:
                body1_name = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, body1_id)
                body2_name = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, body2_id)
                print(f"Collision: {body1_name} ({body1_id}) and {body2_name} ({body2_id})")
                return True
        return False

    def find_collisions_between(self, q_start: ArmPos, q_end: ArmPos, step_size: float) -> bool:
        """
        Determine if there are any collisions between the two given joint positions

        Checks on a linear path between the two in joint space. Checks q_end but not q_start

        Args:
            q_start: starting position
            q_end: ending position
            step_size: gap between checked points (radians)

        Returns:
            True if there is a collision found on the line, False otherwise
        """
        for q in rt.points_between(q_start, q_end, step_size):
            if self.is_in_collision(q):
                return True
        return False

    def _update(self):
        mj.mj_step(self._model, self._data)
        self._view.sync()
        self._time += self._dt

    def step(self, arm_ctrl_sans_bias: ArmPos, gripper_pos: float | None = None):
        """
        Perform a single simulation step. No op if not simulated

        Args:
            arm_ctrl_sans_bias: target arm torque, ignoring the qfrc_bias component
            gripper_pos: target gripper position, or None to not move the gripper
        """
        if not self.is_simulated:
            return
        arm_ctrl = arm_ctrl_sans_bias + self._data.qfrc_bias[self._arm_idxs]
        self._data.ctrl[self._arm_idxs] = arm_ctrl
        if gripper_pos is not None:
            self._data.ctrl[self._gripper_idx] = gripper_pos
        self._update()

    def go_to_position(self, q: ArmPos, duration: float, wait_time: float = 1.):
        """
        Move the arm to the given position

        In simulation, this uses a min jerk path. On the real Franka arm, will use Franka's internal move function

        Args:
            target: the target pose of the gripper
            duration: how long the motion should take (only valid in simulation) (seconds)
            wait_time: how long the motion should pause for after completing the motion (seconds)
        """
        if not self.is_simulated:
            # TO UNCOMMENT
            # self._franka.goto_joints(q)
            if wait_time > 0:
                time.sleep(wait_time)
            return

        q_init = self.q
        nsteps = max(1, int((duration + wait_time) / self._dt))
        start_time = self.time
        for _ in range(nsteps):
            q_des, qd_des = rt.interp_min_jerk(q_init, q, self.time - start_time, duration)
            q_cur = self.q
            qd_cur = self.q_vel
            arm_ctrl = self._kp * (q_des - q_cur) + self._kd * (qd_des - qd_cur)
            self.step(arm_ctrl)

    def go_to_pose(self, target: Pose, duration: float, wait_time: float = 1.):
        """
        Move the gripper to the given target pose

        In simulation, this uses a min jerk path. On the real Franka arm, will use Franka's internal move function

        Args:
            target: the target pose of the gripper
            duration: how long the motion should take (only valid in simulation) (seconds)
            wait_time: how long the motion should pause for after completing the motion (seconds)
        """
        q_goal = self.ik_multi(target, self.q)
        if q_goal is None:
            raise ValueError(f"Could not get to {target}")
        self.go_to_position(q_goal, duration, wait_time)

    def move_gripper(self, gripper_pos: float | bool, duration: float = 0.2):
        """
        Move the gripper to the given position

        Args:
            hand_pos: if float, the position to move the gripper to. If bool, whether to open the gripper. Only bool is
                accepted if not in sim
            duration: how long it should take to move the gripper (this is ignored if not in sim)

        Returns:

        """
        if not self.is_simulated:
            if not isinstance(gripper_pos, bool):
                raise ValueError("Cannot move gripper to a continuous position when not in simulation")
            # TO UNCOMMENT
            # if gripper_pos:
            #     self._franka.open_gripper()
            # else:
            #     self._franka.close_gripper()
            return

        if isinstance(gripper_pos, bool):
            gripper_pos = self.gripper_open if gripper_pos else self.gripper_closed
        start_time = self.time
        arm_ctrl = np.zeros((len(self._arm_idxs),))
        while self.time - start_time < duration:
            self.step(arm_ctrl, gripper_pos)

    def execute_plan(self, sparse_plan: list[ArmPos], wait_time: float = 1., angle_resolution: float = 0.01):
        """
        Move the arm between a list of joint positions

        Args:
            sparse_plan: the list of arm positions to move between
            wait_time: how long the motion should pause for after completing the motion (seconds)
            angle_resolution: if not simulation, most that a single joint can move in one step (radians)
        """
        if not self.is_simulated:
            for pos in sparse_plan:
                self.go_to_position(pos, 0.0, 0.0)
            if wait_time > 0:
                time.sleep(wait_time)

        dense_plan = []
        dense_plan.append(sparse_plan[0])
        for i in range(len(sparse_plan) - 1):
            max_joint_val = np.max(np.abs(sparse_plan[i + 1] - sparse_plan[i]))
            number_of_steps = max(int(np.ceil(max_joint_val / angle_resolution)), 1)
            inc = (sparse_plan[i + 1] - sparse_plan[i]) / number_of_steps
            for j in range(1, number_of_steps + 1):
                step = sparse_plan[i] + j * inc
                dense_plan.append(step)

        for target in dense_plan:
            while np.linalg.norm(target - self.q) > 0.1:
                arm_ctrl = self._kp * (target - self.q) + self._kd * -self.q_vel
                self.step(arm_ctrl)
        nsteps = max(1, int(wait_time / self._dt))
        for _ in range(nsteps):
            self.step(np.zeros((len(self._arm_idxs),)))

    def construct_rrt_plan(self, target_pose: Pose, bias_percent: float = 0.05, connection_step_size: float = 0.1, collision_step_size: float = 0.1, goal_threshold: float = 0.01, max_verts: int = 5000, path_shorten_runs: int = 500) -> list[ArmPos]:
        """
        Create a sparse set of waypoints to move through to get the gripper to a target pose

        Args:
            target_pose: the target position for the gripper
            bias_percent: what percentage of the time to expand direct to goal
            connection_step_size: how big of a step to make when trying to connect to next position (radians)
            collision_step_size: how fine the collision checking should be (radians) (see find_collisions_between)
            goal_threshold: how close to the goal a solution needs to be to be valid (radians)
            max_verts: the maximum number of vertices to add to the RRT before giving up
            path_shorten_runs: how many times we should attempt to shorten the path

        Returns:
            a list of joint positions forming a sparse plan to get to the target pose while avoiding collisions
        """
        # Setup
        verts = [self.q]
        edges = [0]
        q_goal = self.ik_multi(target_pose, self.q)
        if q_goal is None:
            raise ValueError(f"Could not get to {target_pose}")

        # Construct RRT
        found_soln = False
        print("constructing RRT...")
        while len(verts) < max_verts and not found_soln:
            q_rand = self.sample_pos() if np.random.rand() > bias_percent else q_goal
            idx_near = rt.FindNearest(verts, q_rand)
            q_near = verts[idx_near]
            prev_idx = idx_near
            prev_pt = q_near
            for q_connect in rt.points_between(q_near, q_rand, connection_step_size):
                if self.find_collisions_between(prev_pt, q_connect, collision_step_size):
                    break
                edges.append(prev_idx)
                verts.append(q_connect)
                prev_idx = len(verts) - 1
                prev_pt = q_connect
                if (np.linalg.norm(q_goal - q_connect) < goal_threshold and
                        not self.find_collisions_between(q_connect, q_goal, collision_step_size)):
                    found_soln = True
                    edges.append(prev_idx)
                    verts.append(q_goal)
                    break
        if not found_soln:
            raise ValueError(f"Could not plan to {target_pose}")

        # Trace back the path
        vert_idx = -1
        plan = []
        while vert_idx != 0:
            plan.append(verts[vert_idx])
            vert_idx = edges[vert_idx]
        plan.append(verts[0])
        plan = list(reversed(plan))
        return plan

    def go_to_pose_rrt(self, target_pose: Pose, wait_time: float = 1., **rrt_args):
        """
        Move the gripper to a target position using RRT

        Args:
            target_pose: the target pose of the gripper
            wait_time: the amount of time to wait after reaching the target position (seconds)
            **rrt_args: extra arguments to the RRT construction - see construct_rrt_plan
        """
        sparse_plan = self.construct_rrt_plan(target_pose, **rrt_args)
        self.execute_plan(sparse_plan, wait_time)

    def pick_place(self, pick: bool, target_pose: Pose, approach_len: float = 0.1, do_rrt: bool = True):
        """
        Execute a pick or a place

        Args:
            pick: True to execute a pick, False for a place
            target_pose: where the gripper should end up during the pick/place
            approach_len: how long the approach and retreat should be
            do_rrt: whether to perform RRT to get to the inital pose
        """
        approach_pose_t = target_pose.t - target_pose.R[:3, 2] * approach_len
        approach_pose = Pose.from_Rt(target_pose.R, approach_pose_t)
        if do_rrt:
            self.go_to_pose_rrt(approach_pose, wait_time=0.1)
        else:
            self.go_to_pose(approach_pose, duration=5., wait_time=0.1)
        self.move_gripper(pick)
        self.go_to_pose(target_pose, 1.)
        self.move_gripper(not pick)
        self.go_to_pose(approach_pose, 1., wait_time=0.1)


class Environment:
    def __init__(self, root_model_xml_path: str, model_xml_path: str, is_simulated: bool, initial_arm_pose: ArmPos | None = None, blocks: Iterable[Block] = ()):
        model_tree = ET.parse(root_model_xml_path)
        for block in blocks:
            block.add_to_model(model_tree)
        model_tree.write(model_xml_path, encoding="utf-8", xml_declaration=True)

        self.model = mj.MjModel.from_xml_path(model_xml_path)
        self.data = mj.MjData(self.model)

        self.view = mj.viewer.launch_passive(self.model, self.data)
        self.view.cam.distance = 3.0  # Shift camera
        self.view.cam.azimuth += 90

        self._is_simulated = is_simulated
        free_block_idxs = {mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, b.name) for b in blocks if b.free}
        self.arm = Arm(self.model, self.data, self.view, is_simulated, initial_pose=initial_arm_pose, body_idxs_for_collision_to_ignore=free_block_idxs)

    @property
    def is_simulated(self) -> bool:
        return self._is_simulated

    def teardown(self):
        self.view.close()


def grasp_R(approach: Pos3D, closing: Pos3D) -> RotationMatrix:
    """Hand frame: z = approach direction, y = finger closing axis, x = y cross z."""
    # Thank you Anshu
    assert np.isclose(approach @ closing, 0.)
    z = np.asarray(approach, float)
    z /= np.linalg.norm(z)
    y = np.asarray(closing, float)
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def lab3_actions():
    return [
        ActionSchema(
            name="move_vert",
            parameters=("?b", "?s", "?f", "?u", "?d"),
            preconds=(
                # type checking (static)
                ("block", ("?b",)),
                ("src", ("?s",)),
                ("dest", ("?f",)),  # cant do dest u or dest d b/c of null case
                # position checking (static)
                ("up", ("?f", "?u")),
                ("down", ("?f", "?d")),
                # current state checking (dynamic)
                ("clear", ("?f",)),
                ("clear", ("?u",)),
                ("clear", ("?d",)),
                ("at", ("?b", "?s")),
            ),
            add_effects=(
                ("at", ("?b", "?f")),
                ("clear", ("?s",)),
            ),
            del_effects=(
                ("at", ("?b", "?s")),
                ("clear", ("?f",)),
            )
        ),
        ActionSchema(
            name="move_horz",
            parameters=("?b", "?s", "?f", "?l", "?r"),
            preconds=(
                # type checking (static)
                ("block", ("?b",)),
                ("src", ("?s",)),
                ("dest", ("?f",)),  # cant do dest u or dest d b/c of null case
                # position checking (static)
                ("left", ("?f", "?l")),
                ("right", ("?f", "?r")),
                # current state checking (dynamic)
                ("clear", ("?f",)),
                ("clear", ("?l",)),
                ("clear", ("?r",)),
                ("at", ("?b", "?s")),
            ),
            add_effects=(
                ("at", ("?b", "?f")),
                ("clear", ("?s",)),
            ),
            del_effects=(
                ("at", ("?b", "?s")),
                ("clear", ("?f",)),
            )
        )
    ]


def lab3_init():
    return {
        # Blocks
        ("block", ("br0",)),
        ("block", ("br1",)),
        ("block", ("br2",)),
        ("block", ("br3",)),
        ("block", ("bb0",)),
        ("block", ("bb1",)),
        ("block", ("bb2",)),
        ("block", ("bb3",)),
        # Starting positions
        ("src", ("pr0",)),
        ("src", ("pr1",)),
        ("src", ("pr2",)),
        ("src", ("pr3",)),
        ("src", ("pb0",)),
        ("src", ("pb1",)),
        ("src", ("pb2",)),
        ("src", ("pb3",)),
        # Destinations - 00 top left, 20 top right, 02 bottom left
        ("dest", ("pc00",)),
        ("dest", ("pc01",)),
        ("dest", ("pc02",)),
        ("dest", ("pc10",)),
        ("dest", ("pc11",)),
        ("dest", ("pc12",)),
        ("dest", ("pc20",)),
        ("dest", ("pc21",)),
        ("dest", ("pc22",)),
        # Relative positions - up
        ("up", ("pc00", "pc01")),
        ("up", ("pc10", "pc11")),
        ("up", ("pc20", "pc21")),
        ("up", ("pc01", "pc02")),
        ("up", ("pc11", "pc12")),
        ("up", ("pc21", "pc22")),
        ("up", ("pc02", "pnull")),
        ("up", ("pc12", "pnull")),
        ("up", ("pc22", "pnull")),
        # Relative positions - down
        ("down", ("pc00", "pnull")),
        ("down", ("pc10", "pnull")),
        ("down", ("pc20", "pnull")),
        ("down", ("pc01", "pc00")),
        ("down", ("pc11", "pc10")),
        ("down", ("pc21", "pc20")),
        ("down", ("pc02", "pc01")),
        ("down", ("pc12", "pc11")),
        ("down", ("pc22", "pc21")),
        # Relative positions - left
        ("left", ("pc00", "pc10")),
        ("left", ("pc01", "pc11")),
        ("left", ("pc02", "pc12")),
        ("left", ("pc10", "pc20")),
        ("left", ("pc11", "pc21")),
        ("left", ("pc12", "pc22")),
        ("left", ("pc20", "pnull")),
        ("left", ("pc21", "pnull")),
        ("left", ("pc22", "pnull")),
        # Relative positions - right
        ("right", ("pc00", "pnull")),
        ("right", ("pc01", "pnull")),
        ("right", ("pc02", "pnull")),
        ("right", ("pc10", "pc00")),
        ("right", ("pc11", "pc01")),
        ("right", ("pc12", "pc02")),
        ("right", ("pc20", "pc10")),
        ("right", ("pc21", "pc11")),
        ("right", ("pc22", "pc12")),
        # Current position of blocks
        ("at", ("br0", "pr0")),
        ("at", ("br1", "pr1")),
        ("at", ("br2", "pr2")),
        ("at", ("br3", "pr3")),
        ("at", ("bb0", "pb0")),
        ("at", ("bb1", "pb1")),
        ("at", ("bb2", "pb2")),
        ("at", ("bb3", "pb3")),
        # Center is clear
        ("clear", ("pc00",)),
        ("clear", ("pc10",)),
        ("clear", ("pc20",)),
        ("clear", ("pc01",)),
        ("clear", ("pc11",)),
        ("clear", ("pc21",)),
        ("clear", ("pc02",)),
        ("clear", ("pc12",)),
        ("clear", ("pc22",)),
        ("clear", ("pnull",)),
    }


def create_goal(pattern_num: int) -> Set[Predicate]:
    if pattern_num == 1:
        return {
            ("at", ("bb0", "pc00")),
            ("at", ("bb1", "pc10")),
            ("at", ("bb2", "pc20")),
            ("at", ("br0", "pc11")),
            ("at", ("br1", "pc12")),
            ("clear", ("pc01",)),
            ("clear", ("pc02",)),
            ("clear", ("pc21",)),
            ("clear", ("pc22",)),
        }
    elif pattern_num == 2:
        return {
            ("at", ("bb0", "pc00")),
            ("at", ("bb1", "pc10")),
            ("at", ("bb2", "pc01")),
            ("at", ("br0", "pc11")),
            ("at", ("br1", "pc12")),
            ("clear", ("pc02",)),
            ("clear", ("pc20",)),
            ("clear", ("pc21",)),
            ("clear", ("pc22",)),
        }
    elif pattern_num == 3:
        return {
            ("at", ("bb0", "pc00")),
            ("at", ("bb1", "pc20")),
            ("at", ("bb2", "pc02")),
            ("at", ("bb3", "pc22")),
            ("at", ("br0", "pc10")),
            ("at", ("br1", "pc01")),
            ("at", ("br2", "pc12")),
            ("at", ("br3", "pc21")),
            ("clear", ("pc11",)),
        }
    elif pattern_num == 4:
        return {
            ("at", ("bb0", "pc00")),
            ("at", ("bb1", "pc01")),
            ("at", ("bb2", "pc02")),
            ("at", ("br0", "pc10")),
            ("at", ("br1", "pc21")),
            ("at", ("br2", "pc12")),
        }
    else:
        raise ValueError(f"Unknown pattern num {pattern_num}")


def main(pattern_num: int):
    initial_pos = np.array([0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8])

    block_size = 0.02 * np.ones((3,))
    block_spacing = np.array([block_size[0] * 3, 0., 0.])
    red_block_start_pos = np.array([0.25, -0.2, 0.01])
    blue_block_start_pos = np.array([0.25, 0.2, 0.01])
    blocks = []
    name_to_pos = {}
    gripper_offset = np.array([0.0, 0.0, 0.12])
    for idx in range(4):
        red_pos = red_block_start_pos + block_spacing * idx
        blocks.append(Block(
            name=f"RedBlock{idx}",
            pos=red_pos,
            size=block_size,
            rgba=[0.9, 0.2, 0.0, 1.],
            free=True
        ))
        name_to_pos[f"pr{idx}"] = red_pos + gripper_offset

        blue_pos = blue_block_start_pos + block_spacing * idx
        blocks.append(Block(
            name=f"BlueBlock{idx}",
            pos=blue_pos,
            size=block_size,
            rgba=[0.0, 0.2, 0.9, 1.],
            free=True
        ))
        name_to_pos[f"pb{idx}"] = blue_pos + gripper_offset

    center_pos = np.array([0.35, 0.0, 0.0]) + gripper_offset
    x_space = block_spacing
    y_space = np.array([0.0, x_space[0], 0.0])
    for x, y in itertools.product(range(-1, 2), repeat=2):
        name_to_pos[f"pc{x+1}{y+1}"] = center_pos + x * x_space + y * y_space

    planner = Planner(lab3_actions())
    init = lab3_init()
    plan = planner.plan(init, create_goal(pattern_num))
    if plan is None:
        print("No plan found.")
        return
    print("Plan:")

    r_vert = grasp_R(np.array([0., 0., -1.]), np.array([0., 1., 0.]))
    r_horz = grasp_R(np.array([0., 0., -1.]), np.array([1., 0., 0.]))

    env = Environment("franka_emika_panda/panda_torque_table.xml", "franka_emika_panda/panda_torque_table_shelves.xml", True, initial_arm_pose=initial_pos, blocks=blocks)
    try:
        start_pose = env.arm.gripper_pose
        for step, action in enumerate(plan, 1):
            print(f"{step:02d}. {action.label()}")
            pick_pose = Pose.from_Rt(r_vert, name_to_pos[action.args[1]])
            place_r = r_vert if action.name == "move_vert" else r_horz
            place_pose = Pose.from_Rt(place_r, name_to_pos[action.args[2]])
            print("pick...")
            env.arm.pick_place(True, pick_pose, do_rrt=False)
            print("place...")
            env.arm.pick_place(False, place_pose, do_rrt=False)
        env.arm.go_to_pose(start_pose, duration=3., wait_time=1.)
    except:
        raise
    finally:
        env.teardown()


if __name__ == '__main__':
    main(4)
