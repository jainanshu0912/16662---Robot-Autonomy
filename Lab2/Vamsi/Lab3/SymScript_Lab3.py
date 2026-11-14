"""
Lab 3 - Task Planning: rearrange red/blue blocks on a 3x3 grid into the
patterns [pickaxe, axe, box, bow] using SymPlanner (STRIPS + A*).

Symbolic model
--------------
Locations
  * g<r><c>   : the 9 grid cells, r = row (0 = top row of the handout figure),
                c = column (0 = left column of the figure).
  * sL0..sL3  : the 4 red start slots (left of the grid in Figure 1).
  * sR0..sR3  : the 4 blue start slots (right of the grid in Figure 1).
  * off       : a virtual "outside the grid" location; it is always empty and
                is used as the neighbour of edge cells.

Blocks are only distinguished by colour (red / blue).  Same-coloured blocks are
interchangeable, so modelling them by colour removes symmetric states and keeps
A* well below the 50000-expansion limit.

Finger-collision constraint
  The grid cells are close enough that a finger collides with a block in the
  neighbouring cell.  A top grasp is described by the axis the fingers close
  along:
    * axis "left_right": fingers sit in the cells to the left and right
    * axis "up_down":    fingers sit in the cells above and below
  Facts finger_cells(cell, axis, a, b) give the two cells the fingers enter for each
  axis.  A pick OR place at a cell is only allowed with an axis whose two
  neighbour cells are empty.  The start slots are spaced far apart, so they
  have no constraint (both neighbours are `off`) and always use SLOT_AXIS.

Irrecoverable placements
  Because a place needs the same free finger axis as a pick, every block is
  re-graspable at the moment it is put down.  In particular, the robot can
  never drop a block into the centre of a ring of blocks: both axes of the
  centre cell are blocked, so no place action is applicable there.

Predicates
  occ(l, colour), empty(l), handempty(), holding(colour),
  finger_cells(l, axis, a, b)   (static)

Usage
  python SymScript_Lab3.py              # chain the 4 patterns
  python SymScript_Lab3.py --reset      # return to the start layout between patterns
"""
import argparse

import SymPlanner as sp

ROWS, COLS = 3, 3
GRID = [f"g{r}{c}" for r in range(ROWS) for c in range(COLS)]
RED_SLOTS = [f"sL{i}" for i in range(4)]
BLUE_SLOTS = [f"sR{i}" for i in range(4)]
SLOTS = RED_SLOTS + BLUE_SLOTS
OFF = "off"
# Grasp axis used at the start slots: "left_right" = fingers closing left/right (y).
# The slots are lined up front-to-back, so nothing sits beside them, and in the
# Lab 2 scene this is the orientation that is reachable at every slot.
SLOT_AXIS = "left_right"

# Patterns from Figure 2, rows top->bottom. B = blue, R = red, . = empty
PATTERNS = {
    "pickaxe": ["BBB",
                ".R.",
                ".R."],
    "axe":     ["BB.",
                "BR.",
                ".R."],
    "box":     ["BRB",
                "R.R",
                "BRB"],
    "bow":     ["BR.",
                "B.R",
                "BR."],
}
PATTERN_ORDER = ["pickaxe", "axe", "box", "bow"]
COLOUR = {"R": "red", "B": "blue"}


# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
def make_Lab3_domain():
    # Variables (anything starting with "?" is filled in by the planner):
    #   ?loc            where the action happens (grid cell or start slot)
    #   ?colour         colour of the block being moved: "red" or "blue"
    #   ?axis           direction the fingers close along: "left_right" or "up_down"
    #   ?finger_cell_a  the two locations the fingers come down into for
    #   ?finger_cell_b  that ?loc and ?axis (looked up from the finger_cells facts)

    # Top-grasp the ?colour block at ?loc with the fingers closing along ?axis.
    # Both finger cells must be empty.
    pick = sp.ActionSchema(
        name="pick",
        parameters=("?loc", "?colour", "?axis"),
        preconds=(
            ("handempty", ()),
            ("occ", ("?loc", "?colour")),
            ("finger_cells", ("?loc", "?axis", "?finger_cell_a", "?finger_cell_b")),
            ("empty", ("?finger_cell_a",)),
            ("empty", ("?finger_cell_b",)),
        ),
        add_effects=(("holding", ("?colour",)), ("empty", ("?loc",))),
        del_effects=(("handempty", ()), ("occ", ("?loc", "?colour"))),
    )

    # Put the held ?colour block down at the empty location ?loc. Same finger-clearance
    # requirement as pick, so the block is always re-graspable afterwards.
    place = sp.ActionSchema(
        name="place",
        parameters=("?loc", "?colour", "?axis"),
        preconds=(
            ("holding", ("?colour",)),
            ("empty", ("?loc",)),
            ("finger_cells", ("?loc", "?axis", "?finger_cell_a", "?finger_cell_b")),
            ("empty", ("?finger_cell_a",)),
            ("empty", ("?finger_cell_b",)),
        ),
        add_effects=(("occ", ("?loc", "?colour")), ("handempty", ())),
        del_effects=(("holding", ("?colour",)), ("empty", ("?loc",))),
    )
    return [pick, place]


def static_facts():
    facts = {("empty", (OFF,))}  # never deleted: `off` is not a placeable location

    def cell(r, c):
        if 0 <= r < ROWS and 0 <= c < COLS:
            return f"g{r}{c}"
        return OFF

    for r in range(ROWS):
        for c in range(COLS):
            g = f"g{r}{c}"
            facts.add(("finger_cells", (g, "left_right", cell(r, c - 1), cell(r, c + 1))))
            facts.add(("finger_cells", (g, "up_down", cell(r - 1, c), cell(r + 1, c))))
    for s in SLOTS:
        # Start slots have no blocking neighbours, so one grasp axis is enough.
        facts.add(("finger_cells", (s, SLOT_AXIS, OFF, OFF)))
    return facts


# -----------------------------------------------------------------------------
# Problems
# -----------------------------------------------------------------------------
def start_layout():
    """Dynamic facts of Figure 1: 4 red on the left, 4 blue on the right."""
    facts = {("handempty", ())}
    facts |= {("occ", (s, "red")) for s in RED_SLOTS}
    facts |= {("occ", (s, "blue")) for s in BLUE_SLOTS}
    facts |= {("empty", (g,)) for g in GRID}
    return facts


def pattern_goal(rows):
    """Every grid cell must match the pattern (coloured or empty) and the hand
    must be empty. Unused blocks may sit in any start slot."""
    goal = {("handempty", ())}
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            g = f"g{r}{c}"
            goal.add(("occ", (g, COLOUR[ch])) if ch in COLOUR else ("empty", (g,)))
    return goal


def start_goal():
    goal = {("handempty", ())}
    goal |= {("occ", (s, "red")) for s in RED_SLOTS}
    goal |= {("occ", (s, "blue")) for s in BLUE_SLOTS}
    return goal


def colour_counts(state):
    counts = {"red": 0, "blue": 0}
    for name, args in state:
        if name == "occ":
            counts[args[1]] += 1
        elif name == "holding":
            counts[args[0]] += 1
    return counts


def graspable_axis(cell, occupied):
    """Return an axis whose finger cells are both free, or None."""
    r, c = int(cell[1]), int(cell[2])

    def free(rr, cc):
        return not (0 <= rr < ROWS and 0 <= cc < COLS) or f"g{rr}{cc}" not in occupied

    if free(r, c - 1) and free(r, c + 1):
        return "left_right"
    if free(r - 1, c) and free(r + 1, c):
        return "up_down"
    return None


def quick_infeasible(state, goal):
    """Exact feasibility check before search.

    1. The goal must not need more blocks of a colour than exist.
    2. Pick and place have identical clearance conditions, so moves are
       reversible and every legal grid layout is connected to the empty grid.
       Therefore the goal layout is reachable iff it can be disassembled one
       block at a time (removing blocks only frees space, so greedy removal is
       exact). If the disassembly gets stuck, the remaining blocks form a
       cluster where the last block placed would always hit a neighbour.
    """
    have = colour_counts(state)
    need = {"red": 0, "blue": 0}
    occupied = set()
    for name, args in goal:
        if name == "occ":
            need[args[1]] += 1
            if args[0] in GRID:
                occupied.add(args[0])
    for col in need:
        if need[col] > have[col]:
            return f"needs {need[col]} {col} blocks but only {have[col]} exist"

    while occupied:
        removable = [g for g in sorted(occupied) if graspable_axis(g, occupied)]
        if not removable:
            return ("cells " + ", ".join(sorted(occupied)) +
                    " can never be filled: the last block placed there would "
                    "collide with a neighbour on both finger axes")
        occupied -= set(removable)
    return None


class Lab3Planner(sp.Planner):
    """SymPlanner's A* with two speed-ups (SymPlanner.py is left unchanged):
      * facts are indexed by predicate name when unifying preconditions
      * weighted goal-count heuristic (each unmet cell goal costs ~2 actions:
        a pick and a place), which keeps the search well under the budget.
    """
    H_WEIGHT = 2

    def applicable_actions(self, state):
        by_name = {}
        for f in state:
            by_name.setdefault(f[0], []).append(f)
        out = []
        for schema in self.actions:
            pre = schema.preconds

            def backtrack(i, theta):
                if i == len(pre):
                    out.append(schema.ground(theta))
                    return
                p = sp.substitute(pre[i], theta)
                for f in by_name.get(p[0], ()):
                    t = sp.unify(p, f, theta)
                    if t is not None:
                        backtrack(i + 1, t)

            backtrack(0, {})
        return out

    def heuristic(self, state, goal):
        return self.H_WEIGHT * sum(1 for g in goal if g not in state)


# -----------------------------------------------------------------------------
# Plan -> robot primitives
# -----------------------------------------------------------------------------
# Geometry (metres, robot base frame). These values are the ones validated in
# the Lab 2 MuJoCo scene by Lab3_Sim.py; re-measure them on the real table.
# Rows of the figure run along robot -x (row 0 is farthest from the robot),
# columns run along robot -y (column 0 on the robot's left, +y).
BLOCK_HALF = 0.02            # 4 cm cubes, as in Lab 2
GRID_CENTER = (0.46, 0.00)   # centre cell g11
GRID_PITCH = 0.06            # 2 cm gap between blocks: an open finger does not fit
SLOT_X = [0.58, 0.50, 0.42, 0.34]   # start slots, 8 cm apart along x
# Further out than 0.155 the hand hits the shelves in the Lab 2 collision model.
SLOT_Y_RED, SLOT_Y_BLUE = 0.15, -0.15
BLOCK_Z = BLOCK_HALF         # grasp height = block centre

# World direction the fingers close along for each symbolic axis.
#   "left_right" -> fingers in the left/right neighbour cells -> close along world y
#   "up_down" -> fingers in the front/back neighbour cells -> close along world x
AXIS_FINGER_DIR = {"left_right": (0.0, 1.0, 0.0), "up_down": (1.0, 0.0, 0.0)}
# Equivalent end-effector yaw (Franka fingers close along the hand y-axis);
# check on the robot before running.
AXIS_YAW = {"left_right": 0.0, "up_down": 1.5708}


def location_xyz(loc):
    if loc.startswith("g"):
        r, c = int(loc[1]), int(loc[2])
        x = GRID_CENTER[0] - (r - 1) * GRID_PITCH
        y = GRID_CENTER[1] + (1 - c) * GRID_PITCH
        return (x, y, BLOCK_Z)
    i = int(loc[2])
    return (SLOT_X[i], SLOT_Y_RED if loc.startswith("sL") else SLOT_Y_BLUE, BLOCK_Z)


def plan_to_primitives(plan):
    """Turn a symbolic plan into [(action, colour, (x, y, z), yaw), ...] that a
    pick/place controller (e.g. from Lab 2) can execute in order."""
    prims = []
    for act in plan:
        loc, col, ax = act.args
        prims.append((act.name, col, location_xyz(loc), AXIS_YAW[ax]))
    return prims


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def print_grid(state):
    sym = {"red": "R", "blue": "B"}
    occ = {args[0]: sym[args[1]] for name, args in state if name == "occ"}
    for r in range(ROWS):
        print("      " + " ".join(occ.get(f"g{r}{c}", ".") for c in range(COLS)))
    left = "".join(occ.get(s, ".") for s in RED_SLOTS)
    right = "".join(occ.get(s, ".") for s in BLUE_SLOTS)
    print(f"      left slots [{left}]  right slots [{right}]")


def solve(planner, statics, dyn_state, goal, label):
    """Plan from dyn_state to goal. Returns (plan, new_dyn_state) or (None, dyn_state)."""
    reason = quick_infeasible(dyn_state, goal)
    if reason:
        print(f"[{label}] infeasible ({reason}) -> skipping")
        return None, dyn_state

    plan = planner.plan(dyn_state | statics, goal)
    if plan is None:
        print(f"[{label}] no plan found -> skipping")
        return None, dyn_state

    state = set(dyn_state)
    for act in plan:
        state = planner.apply(state, act)
    print(f"[{label}] plan with {len(plan)} actions:")
    for step, act in enumerate(plan, 1):
        print(f"   {step:02d}. {act.label()}")
    print_grid(state)
    return plan, state


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true",
                    help="return all blocks to the start layout between patterns")
    ap.add_argument("--primitives", action="store_true",
                    help="also print the (action, colour, xyz, yaw) primitives")
    args = ap.parse_args()

    planner = Lab3Planner(make_Lab3_domain())
    statics = static_facts()
    state = start_layout()

    print("Start layout:")
    print_grid(state)

    full_plan = []
    for name in PATTERN_ORDER:
        if args.reset and full_plan:
            plan, state = solve(planner, statics, state, start_goal(), f"reset before {name}")
            full_plan += plan or []
        plan, state = solve(planner, statics, state, pattern_goal(PATTERNS[name]), name)
        full_plan += plan or []

    print(f"\nTotal actions: {len(full_plan)}")
    if args.primitives:
        print("Robot primitives (action, colour, xyz, yaw):")
        for p in plan_to_primitives(full_plan):
            name, col, (x, y, z), yaw = p
            print(f"   {name:5s} {col:4s} ({x:.3f}, {y:.3f}, {z:.3f}) yaw={yaw:.3f}")


if __name__ == "__main__":
    main()
