"""
Lab 3 in simulation: task plan (SymScript_Lab3) -> motion plan + execution
with the Lab 2 stack (MotionPlanner, TaskPlanner step format, SimExecutor).

The planning loop is Lab3Runner in Lab3_Shared.py (also used by
Lab3_Robot.py). This file adds the MuJoCo side: the scene (Lab 2 table +
shelves with 8 blocks), Lab 2's SimExecutor, and a check after every action
that the moved block is where it should be and no other block was disturbed.

Usage
  python Lab3_Sim.py                 # with viewer, patterns chained
  python Lab3_Sim.py --headless      # no viewer (faster)
  python Lab3_Sim.py --reset         # reset to the start layout between patterns
  LAB2_DIR=/path/to/Lab2 python Lab3_Sim.py   # if Lab 2 lives elsewhere
"""
import os
import time
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import mujoco as mj

HERE = os.path.dirname(os.path.abspath(__file__))

import SymScript_Lab3 as L                                    # noqa: E402
from Lab3_Shared import LAB2_DIR, BLOCKS, Q_HOME, GRIPPER_OPEN, Lab3Runner   # noqa: E402
from SimpleScript import SimExecutor   # noqa: E402  (Lab 2; on sys.path via Lab3_Shared)

ROOT_MODEL_XML = os.path.join(LAB2_DIR, "franka_emika_panda", "panda_torque_table.xml")
MODEL_XML = os.path.join(HERE, "lab3_scene.xml")

POS_TOL = 0.012         # a block counts as "at" a location within this xy distance
RGBA = {"red": [0.9, 0.15, 0.1, 1], "blue": [0.05, 0.4, 0.8, 1]}


# ----------------------------------------------------------------------------- scene
def build_scene():
    """Lab 2 table + shelves, grid markers and the 8 blocks. Returns block names by slot."""
    tree = ET.parse(ROOT_MODEL_XML)
    root = tree.getroot()
    root.find("compiler").set("meshdir", os.path.join(LAB2_DIR, "franka_emika_panda", "assets"))
    world = root.find("worldbody")

    def box(name, pos, half, rgba, free=False, collide=True):
        body = ET.SubElement(world, "body", {"name": name, "pos": "%g %g %g" % tuple(pos)})
        attrs = {"type": "box", "size": "%g %g %g" % tuple(half), "rgba": "%g %g %g %g" % tuple(rgba),
                 "density": "20"}
        if not collide:
            attrs.update({"contype": "0", "conaffinity": "0"})
        ET.SubElement(body, "geom", attrs)
        if free:
            ET.SubElement(body, "freejoint")

    for name, pos, half in BLOCKS:
        box(name, pos, half, [0.2, 0.2, 0.9, 1])
    for g in L.GRID:  # visual-only markers for the 3x3 grid
        x, y, _ = L.location_xyz(g)
        box("mark_" + g, [x, y, 0.0006], [0.026, 0.026, 0.0005], [0.75, 0.75, 0.75, 1], collide=False)

    block_at = {}
    for slot in L.SLOTS:
        colour = "red" if slot in L.RED_SLOTS else "blue"
        name = "%s_%s" % (colour, slot[-1])
        box(name, L.location_xyz(slot), [L.BLOCK_HALF]*3, RGBA[colour], free=True)
        block_at[slot] = name
    tree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)
    return block_at


# ----------------------------------------------------------------------------- checks
class SceneChecker:
    """Compares the simulated blocks with what the symbolic plan expects."""

    def __init__(self, model, data, block_at):
        self.model, self.data = model, data
        self.block_at = dict(block_at)   # location -> block name (expected)
        self.held = None
        self.max_err = 0.0

    def pos(self, name):
        return self.data.body(name).xpos.copy()

    def check_resting(self, label):
        for loc, name in self.block_at.items():
            p, target = self.pos(name), np.array(L.location_xyz(loc))
            err = np.linalg.norm(p[:2] - target[:2])
            self.max_err = max(self.max_err, err)
            if err > POS_TOL or abs(p[2] - L.BLOCK_HALF) > 0.005:
                raise RuntimeError("%s: block %s should rest at %s %s but is at %s (xy error %.1f mm)"
                                   % (label, name, loc, np.round(target, 3), np.round(p, 3), 1e3*err))

    def after(self, act):
        loc, colour, _ = act.args
        if act.name == "pick":
            self.held = self.block_at.pop(loc)
            assert self.held.startswith(colour)
            if self.pos(self.held)[2] < 0.08:
                raise RuntimeError("%s: block %s was not lifted" % (act.label(), self.held))
        else:
            self.block_at[loc], self.held = self.held, None
        self.check_resting(act.label())

    def grid_string(self):
        rows = []
        for r in range(L.ROWS):
            rows.append("".join(self.block_at.get("g%d%d" % (r, c), ".")[0].upper().replace(".", ".")
                                for c in range(L.COLS)))
        return rows


def snapshot(model, data, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = mj.Renderer(model, 480, 640)
        cam = mj.MjvCamera()
        cam.lookat[:] = [L.GRID_CENTER[0], 0, 0.02]
        cam.distance, cam.azimuth, cam.elevation = 0.9, 180, -65
        r.update_scene(data, cam)
        plt.imsave(path, r.render())
        r.close()
        print("   saved", path)
    except Exception as e:  # rendering is optional (needs an OpenGL context)
        print("   (snapshot skipped: %s)" % e)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="run without the viewer")
    ap.add_argument("--reset", action="store_true", help="reset to the start layout between patterns")
    ap.add_argument("--snapshots", action="store_true", help="save a picture after each pattern")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    block_at = build_scene()
    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)
    data.qpos[:7] = Q_HOME
    data.qpos[7:9] = GRIPPER_OPEN
    mj.mj_forward(model, data)

    v = None
    if not args.headless:
        from mujoco import viewer
        v = viewer.launch_passive(model, data)
        v.cam.distance = 2.0
        v.cam.azimuth += 90

    ex = SimExecutor(model, data, v)
    checker = SceneChecker(model, data, block_at)

    def after_pattern(name):
        assert checker.grid_string() == L.PATTERNS[name], (checker.grid_string(), L.PATTERNS[name])
        print("   simulated grid matches the %s pattern (every action checked)" % name)
        if args.snapshots:
            snapshot(model, data, os.path.join(HERE, "lab3_%s.png" % name))

    runner = Lab3Runner(ex, q_start=Q_HOME, seed=args.seed,
                        after_action=checker.after, after_pattern=after_pattern)
    try:
        ex.Hold(Q_HOME, 1.0)
        checker.check_resting("start layout")
        wall = time.time()
        runner.run_all(reset=args.reset)
        runner.go_home()
        runner.print_summary("built and verified in sim")
        print("   worst block placement error: %.1f mm" % (1e3*checker.max_err))
        print("   sim time %.0f s, wall time %.0f s" % (data.time, time.time() - wall))
        if v is not None:
            ex.Hold(Q_HOME, 3.0)
    finally:
        if v is not None:
            v.close()


if __name__ == "__main__":
    main()
