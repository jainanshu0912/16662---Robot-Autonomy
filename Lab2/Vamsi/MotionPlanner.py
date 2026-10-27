import random
import numpy as np

import RobotUtil as rt
import Franka


# Gripper (TCP) frame expressed in the Franka FK end-effector frame. The FK end effector is the
# MuJoCo "hand" origin; the fingers are rotated -45 deg about z and the pads sit 0.103m out.
# TCP frame: z = approach direction, y = finger closing direction.
T_EE_TCP = rt.rpyxyz2H([0., 0., -np.pi/4], [0., 0., 0.103])
T_TCP_EE = np.linalg.inv(T_EE_TCP)


def GraspFrame(pos, approach, finger_axis):
    """Homogeneous TCP pose from a position, approach direction (z) and finger closing axis (y)."""
    z = np.asarray(approach, dtype=float); z /= np.linalg.norm(z)
    y = np.asarray(finger_axis, dtype=float); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
    return T


class CollisionEnv:
    """Obstacle boxes in the (points, axes) format used by Franka.DetectCollision."""

    def __init__(self, blocks, margin=0.0):
        # blocks: [name, center, half_sizes] as in SimpleScript BLOCKS
        self.names = []
        self.points = []
        self.axes = []
        for name, pos, half in blocks:
            pts, axs = rt.BlockDesc2Points(rt.rpyxyz2H([0., 0., 0.], pos), [2*h + 2*margin for h in half])
            self.names.append(name)
            self.points.append(pts)
            self.axes.append(axs)

    def Subset(self, ignore=()):
        # Ignoring a name also ignores its sub-parts, e.g. "LShelfBottom" -> "LShelfBottomSupp1"
        idx = [i for i, n in enumerate(self.names) if not any(n.startswith(g) for g in ignore)]
        return [self.points[i] for i in idx], [self.axes[i] for i in idx]


class MotionPlanner:
    def __init__(self, env, seed=0):
        self.arm = Franka.FrankArm()
        self.env = env
        self.qmin = np.array(self.arm.qmin)
        self.qmax = np.array(self.arm.qmax)
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------ kinematics
    def FK(self, q):
        T, J = self.arm.ForwardKin(list(q))
        return np.array(T[-1]), np.array(J)

    def TCPPose(self, q):
        return self.FK(q)[0] @ T_EE_TCP

    def IK(self, T_tcp, q_seed, pos_tol=1e-3, rot_tol=1e-2):
        """IK for a TCP pose using FrankArm.IterInvKin (HW1). Returns q or None."""
        T_goal = np.asarray(T_tcp) @ T_TCP_EE
        q_seed = np.clip(np.array(q_seed, dtype=float), self.qmin, self.qmax)
        q, err = self.arm.IterInvKin(list(q_seed), T_goal, x_eps=pos_tol, r_eps=rot_tol)
        q, err = np.array(q, dtype=float), np.array(err)
        # IterInvKin stops after a fixed iteration count and ignores joint limits
        if np.linalg.norm(err[:3]) > pos_tol or np.linalg.norm(err[3:]) > rot_tol:
            return None
        if np.any(q < self.qmin) or np.any(q > self.qmax):
            return None
        return q

    def IKSolutions(self, T_tcp, q_seeds, n_random=10, ignore=()):
        """Collision-free IK solutions from several seeds, sorted by distance to the first seed."""
        seeds = [np.asarray(s, dtype=float) for s in q_seeds]
        seeds += [np.array(self.arm.SampleRobotConfig()) for _ in range(n_random)]
        sols = []
        for s in seeds:
            q = self.IK(T_tcp, s)
            if q is None or self.InCollision(q, ignore):
                continue
            if all(np.linalg.norm(q - p) > 1e-2 for p in sols):
                sols.append(q)
        sols.sort(key=lambda q: np.linalg.norm(q - seeds[0]))
        return sols

    # ------------------------------------------------------------------ collision checking
    def InCollision(self, q, ignore=()):
        pts, axs = self.env.Subset(ignore)
        return self.arm.DetectCollision(list(q), pts, axs)

    def EdgeInCollision(self, q1, q2, ignore=()):
        pts, axs = self.env.Subset(ignore)
        return self.arm.DetectCollisionEdge(list(q1), list(q2), pts, axs)

    # ------------------------------------------------------------------ RRT (from HW2 RRTQuery)
    def RRTQuery(self, qInit, qGoal, max_vertices=3000, step_size=0.3, goal_bias=0.1,
                 connect_dist=1.0, shortcut_iters=150):
        qInit, qGoal = np.array(qInit, dtype=float), np.array(qGoal, dtype=float)
        if self.InCollision(qInit) or self.InCollision(qGoal):
            print("RRT: start or goal in collision")
            return None
        if not self.EdgeInCollision(qInit, qGoal):
            return [qInit, qGoal]

        rrtVertices = [qInit]  # list of vertices
        rrtEdges = [0]  # parent of each vertex
        FoundSolution = False

        while len(rrtVertices) < max_vertices and not FoundSolution:
            qRand = qGoal if np.random.rand() < goal_bias else np.array(self.arm.SampleRobotConfig())

            idx = rt.FindNearest(rrtVertices, qRand)
            qNear = rrtVertices[idx]
            d = qRand - qNear
            dist = np.linalg.norm(d)
            if dist < 1e-9:
                continue
            qNew = qRand if dist <= step_size else qNear + d/dist*step_size

            if self.InCollision(qNew) or self.EdgeInCollision(qNear, qNew):
                continue

            rrtVertices.append(qNew)
            rrtEdges.append(idx)

            # Try to connect straight to the goal once we are close enough
            if np.linalg.norm(qNew - qGoal) < connect_dist and not self.EdgeInCollision(qNew, qGoal):
                rrtVertices.append(qGoal)
                rrtEdges.append(len(rrtVertices) - 2)
                FoundSolution = True

        if not FoundSolution:
            print("RRT: no solution found")
            return None
        print("RRT found goal with", len(rrtVertices), "vertices")

        # Extract path
        c = len(rrtVertices) - 1
        plan = [rrtVertices[c]]
        while c != 0:
            c = rrtEdges[c]
            plan.insert(0, rrtVertices[c])

        # Path shortening
        for _ in range(shortcut_iters):
            if len(plan) < 3:
                break
            a, b = sorted(np.random.choice(len(plan), 2, replace=False))
            if b - a < 2:
                continue
            if not self.EdgeInCollision(plan[a], plan[b]):
                plan = plan[:a + 1] + plan[b:]
        return plan

    def PlanToConfig(self, qInit, qGoal, attempts=3):
        for _ in range(attempts):
            plan = self.RRTQuery(qInit, qGoal)
            if plan is not None:
                return plan
        raise RuntimeError("Motion planning failed")

    # ------------------------------------------------------------------ Cartesian straight-line moves
    def CartesianPath(self, q_start, T_goal, step=0.01, ignore=()):
        """Joint path moving the TCP in a straight line (fixed orientation) to T_goal.
        Obstacles in `ignore` (e.g. the support surface being placed on) are not checked."""
        T_start = self.TCPPose(q_start)
        n = max(2, int(np.ceil(np.linalg.norm(T_goal[:3, 3] - T_start[:3, 3])/step)) + 1)
        path = [np.array(q_start, dtype=float)]
        for s in np.linspace(0, 1, n)[1:]:
            T = np.array(T_goal, dtype=float)
            T[:3, 3] = T_start[:3, 3] + s*(T_goal[:3, 3] - T_start[:3, 3])
            q = self.IK(T, path[-1])
            if q is None or np.max(np.abs(q - path[-1])) > 0.2:
                raise RuntimeError("Cartesian path: IK failed / discontinuous")
            if self.InCollision(q, ignore):
                raise RuntimeError("Cartesian path in collision")
            path.append(q)
        return path
