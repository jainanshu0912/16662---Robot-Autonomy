import numpy as np, mujoco as mj
from mujoco import viewer
import RobotUtil as rt
from scene import build_scene

KP = np.array([120,120,100,90,60,40,30], float)
KD = np.array([  8,  8,  6, 5, 4, 3, 2], float)
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0125
TCP_OFFSET = 0.1034          # hand origin -> fingertip centre along hand z (tune this)
HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])


class Sim:
    def __init__(self, show=True):
        self.model = mj.MjModel.from_xml_path(build_scene())
        self.data  = mj.MjData(self.model)
        self.scratch = mj.MjData(self.model)          # for IK/collision, never stepped
        self.hid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hand")
        self.bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "Block")
        self.lo = self.model.jnt_range[:7,0]; self.hi = self.model.jnt_range[:7,1]
        self.data.qpos[:7] = HOME; self.data.qpos[7:9] = GRIP_OPEN
        self.grip = GRIP_OPEN
        mj.mj_forward(self.model, self.data)
        self.v = viewer.launch_passive(self.model, self.data) if show else None
        if self.v: self.v.cam.distance = 3.0; self.v.cam.azimuth += 90

    # ---------- execution ----------
    def q(self): return self.data.qpos[:7].copy()

    def _step(self, q_des, qd_des):
        d = self.data
        tau = KP*(q_des-d.qpos[:7]) + KD*(qd_des-d.qvel[:7])
        d.ctrl[:7] = tau + d.qfrc_bias[:7]
        d.ctrl[7]  = self.grip
        mj.mj_step(self.model, d)
        if self.v: self.v.sync()

    def move_path(self, path, speed=0.8):
        """path: list of 7-vectors. Each segment uses a min-jerk profile."""
        dt = self.model.opt.timestep
        for a, b in zip(path[:-1], path[1:]):
            T = max(0.6, np.linalg.norm(b-a)/speed)
            for k in range(int(T/dt)):
                qd, qdd = rt.interp_min_jerk(a, b, k*dt, T)
                self._step(qd, qdd)
        for _ in range(int(0.3/dt)):                   # settle at the last point
            self._step(path[-1], np.zeros(7))

    def set_gripper(self, g, T=1.0):
        self.grip = g
        qh = self.q()
        for _ in range(int(T/self.model.opt.timestep)):
            self._step(qh, np.zeros(7))

    # ---------- kinematics ----------
    def tcp(self, d):
        R = d.xmat[self.hid].reshape(3,3)
        return d.xpos[self.hid] + R[:,2]*TCP_OFFSET, R

    def ik(self, pos, R, q0, iters=300, lam=0.05):
        d = self.scratch; q = q0.copy()
        d.qpos[7:9] = GRIP_OPEN
        for _ in range(iters):
            d.qpos[:7] = q; mj.mj_forward(self.model, d)
            p, Rc = self.tcp(d)
            e_p = pos - p
            e_r = 0.5*sum(np.cross(Rc[:,i], R[:,i]) for i in range(3))   # orientation error
            e = np.r_[e_p, e_r]
            if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 1e-2: return q
            jp = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
            mj.mj_jac(self.model, d, jp, jr, p, self.hid)
            J = np.vstack([jp, jr])[:, :7]
            dq = J.T @ np.linalg.solve(J@J.T + lam**2*np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return None

    def ik_multi(self, pos, R, q0, tries=15):
        for i in range(tries):
            seed = q0 if i == 0 else np.random.uniform(self.lo, self.hi)
            q = self.ik(pos, R, seed)
            if q is not None and not self.collides(q): return q
        return None

    # ---------- collision ----------
    def collides(self, q):
        d = self.scratch
        d.qpos[:7] = q; d.qpos[7:9] = self.grip if self.grip > 0.02 else GRIP_OPEN
        mj.mj_forward(self.model, d)
        for c in d.contact[:d.ncon]:
            b1 = self.model.geom_bodyid[c.geom1]; b2 = self.model.geom_bodyid[c.geom2]
            if self.bid in (b1, b2): continue          # block is handled by the grasp logic
            return True
        return False

    def edge_free(self, a, b, res=0.05):
        n = max(2, int(np.linalg.norm(b-a)/res))
        return all(not self.collides(a + (b-a)*s) for s in np.linspace(0,1,n))

    # ---------- RRT-Connect ----------
    def plan(self, q0, q1, step=0.2, iters=4000):
        if self.collides(q1): return None
        Ta, Tb = [q0], [q1]; Pa, Pb = [-1], [-1]; swapped = False
        def extend(T, P, q):
            i = int(np.argmin([np.linalg.norm(q-n) for n in T]))
            dv = q - T[i]; L = np.linalg.norm(dv)
            new = q if L <= step else T[i] + dv/L*step
            if self.edge_free(T[i], new): T.append(new); P.append(i); return len(T)-1
            return None
        for _ in range(iters):
            qr = np.random.uniform(self.lo, self.hi)
            ia = extend(Ta, Pa, qr)
            if ia is not None:
                while True:                             # greedily connect other tree
                    ib = extend(Tb, Pb, Ta[ia])
                    if ib is None: break
                    if np.allclose(Tb[ib], Ta[ia]):
                        pa, i = [], ia
                        while i != -1: pa.append(Ta[i]); i = Pa[i]
                        pb, i = [], ib
                        while i != -1: pb.append(Tb[i]); i = Pb[i]
                        path = pa[::-1] + pb
                        if swapped: path = path[::-1]
                        return self.shortcut(path)
            Ta, Tb, Pa, Pb = Tb, Ta, Pb, Pa; swapped = not swapped
        return None

    def shortcut(self, path, n=60):
        path = list(path)
        for _ in range(n):
            if len(path) < 3: break
            i, j = sorted(np.random.randint(0, len(path), 2))
            if j - i > 1 and self.edge_free(path[i], path[j]):
                path = path[:i+1] + path[j:]
        return path

def grasp_R(approach, closing):
#"""Hand frame: z = approach direction, y = finger closing axis, x = y cross z."""
    z = np.asarray(approach, float); z /= np.linalg.norm(z)
    y = np.asarray(closing, float);  y = y - z*(y@z); y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])

R_TOP  = grasp_R(approach=[0,0,-1], closing=[0,1,0])   # pointing down
R_SIDE = grasp_R(approach=[1,0, 0], closing=[0,1,0])   # horizontal, pointing away from robot
R_SIDE_L = grasp_R(approach=[0, 1, 0], closing=[1, 0, 0])
R_SIDE_R = grasp_R([0, -1, 0], [1, 0, 0]) 

def go_to(sim, pos, R, tag=""):
    """Collision-free plan to a pose."""
    q = sim.ik_multi(pos, R, sim.q())
    assert q is not None, f"IK failed: {tag}"
    path = sim.plan(sim.q(), q)
    assert path is not None, f"Plan failed: {tag}"
    sim.move_path(path)

def move_linear(sim, pos0, pos1, R, n=8):
    """Straight Cartesian line (approach/retreat), seeded IK each step."""
    q = sim.q(); path = [q]
    for s in np.linspace(0, 1, n)[1:]:
        q = sim.ik(pos0 + (pos1-pos0)*s, R, q)
        assert q is not None, "linear IK failed"
        path.append(q)
    sim.move_path(path, speed=0.5)

def pick(sim, pos, R, standoff=0.12):
    pos = np.asarray(pos, float); pre = pos - R[:,2]*standoff     # back off along approach
    sim.set_gripper(GRIP_OPEN, 0.5)
    go_to(sim, pre, R, "pre-grasp")
    move_linear(sim, pre, pos, R)
    sim.set_gripper(GRIP_CLOSED)
    move_linear(sim, pos, pos + np.array([0,0,0.10]) if R is R_TOP else pre, R)   # lift / retreat

def place(sim, pos, R, standoff=0.12):
    pos = np.asarray(pos, float)
    pre = pos - R[:,2]*standoff
    go_to(sim, pre, R, "pre-place")
    move_linear(sim, pre, pos, R)
    sim.set_gripper(GRIP_OPEN)
    move_linear(sim, pos, pre, R)

SHELF_Z = {"bottom": 0.1475, "middle": 0.3475, "top": 0.5475}   # shelf board top surfaces
HALF = 0.02                                                      # block half-size
BOARD_Y_EDGE = 0.279                                             # inner edge of each board (|y|)

def run(left_xy, right_level, right_xy):
    sim = Sim()
    blk = sim.data.xpos[sim.bid].copy()                          # block starts on table
    tcp_h = lambda p: np.array([p[0], p[1], p[2]])

    # 1. top-down pick from the table
    pick(sim, [blk[0], blk[1], 0.03], R_TOP)

    # 2. top-down place on the left shelf, near the edge
    zL = SHELF_Z["bottom"] + HALF
    place(sim, [left_xy[0], left_xy[1], zL + 0.01], R_TOP)

    # 3. regrasp: side grasp on the left shelf
    #pick(sim, [left_xy[0], left_xy[1], zL], R_SIDE)
    pick(sim, [left_xy[0], left_xy[1], zL], R_SIDE_L)

    # 4. place on a right shelf
    zR = SHELF_Z[right_level] + HALF
    #place(sim, [right_xy[0], right_xy[1], zR + 0.01], R_SIDE)
    place(sim, [right_xy[0], right_xy[1], zR + 0.01], R_SIDE_R)

    # 5. return home
    path = sim.plan(sim.q(), HOME)
    sim.move_path(path)
    sim.set_gripper(GRIP_OPEN, 0.5)

if __name__ == "__main__":
    np.random.seed(0)
    run(left_xy=(0.35, 0.30), right_level="top", right_xy=(0.35, -0.32))
