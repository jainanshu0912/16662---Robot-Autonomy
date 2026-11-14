"""
Symbolic domain for Lab 3: rearranging red/blue blocks on a 3x3 grid.

Locations
  - Grid cells  g{r}{c}, r = row (0 = far from robot), c = column (0 = robot's left)
  - Home slots  hr0..hr3 (red side, robot's left) and hb0..hb3 (blue side, robot's right)
  - "void"      a virtual, always-empty location used as the neighbour of edge cells

Predicates
  empty(?l)           location ?l is free
  colored(?l, ?c)     location ?l holds a block of colour ?c
  (Blocks of the same colour are interchangeable, so the planner only reasons about colours.
   This removes a huge amount of symmetry from the search. The executor keeps track of which
   physical block sits where.)
  accepts(?l, ?c)     a block of colour ?c may be put at ?l        (static: grid cells take
                      both colours, home slots only take their own colour)
  xnbr(?l, ?n1, ?n2)  neighbours of ?l along the robot's x axis    (static)
  ynbr(?l, ?n1, ?n2)  neighbours of ?l along the robot's y axis    (static)

Grasp constraint
  The grid is tight enough that an open finger collides with a block in a neighbouring cell.
  A top grasp whose fingers close along axis A is therefore only allowed when both
  neighbours of the cell along A are empty. The same rule applies when placing, which also
  stops the robot from dropping a block into the middle of a ring of blocks.
  There are four move schemas, one for each (pick axis, place axis) combination; the axes
  are encoded in the action name so the executor knows how to orient the gripper.
"""
import SymPlanner as sp

ROWS, COLS = 3, 3
RED_BLOCKS = ["r0", "r1", "r2", "r3"]
BLUE_BLOCKS = ["b0", "b1", "b2", "b3"]
RED_HOME = ["hr0", "hr1", "hr2", "hr3"]
BLUE_HOME = ["hb0", "hb1", "hb2", "hb3"]
VOID = "void"


def cell(r, c):
    return f"g{r}{c}"


GRID_CELLS = [cell(r, c) for r in range(ROWS) for c in range(COLS)]


def _cell_or_void(r, c):
    return cell(r, c) if 0 <= r < ROWS and 0 <= c < COLS else VOID


def make_Grid_domain():
    schemas = []
    for pick_axis in ("x", "y"):
        for place_axis in ("x", "y"):
            schemas.append(sp.ActionSchema(
                name=f"move-{pick_axis}{place_axis}",
                parameters=("?from", "?to", "?c"),
                preconds=(
                    ("colored", ("?from", "?c")),
                    (f"{pick_axis}nbr", ("?from", "?p1", "?p2")),
                    ("empty", ("?p1",)),
                    ("empty", ("?p2",)),
                    ("accepts", ("?to", "?c")),
                    ("empty", ("?to",)),
                    (f"{place_axis}nbr", ("?to", "?q1", "?q2")),
                    ("empty", ("?q1",)),
                    ("empty", ("?q2",)),
                ),
                add_effects=(
                    ("empty", ("?from",)),
                    ("colored", ("?to", "?c")),
                ),
                del_effects=(
                    ("empty", ("?to",)),
                    ("colored", ("?from", "?c")),
                ),
            ))
    return schemas


def static_facts():
    facts = {("empty", (VOID,))}
    # Grid cells: x axis = rows (towards/away from robot), y axis = columns (left/right)
    for r in range(ROWS):
        for c in range(COLS):
            l = cell(r, c)
            facts.add(("accepts", (l, "red")))
            facts.add(("accepts", (l, "blue")))
            facts.add(("xnbr", (l, _cell_or_void(r - 1, c), _cell_or_void(r + 1, c))))
            facts.add(("ynbr", (l, _cell_or_void(r, c - 1), _cell_or_void(r, c + 1))))
    # Home slots are 10 cm apart, far enough that open fingers never reach a neighbour,
    # so they can be grasped along either axis.
    for home, colour in ((RED_HOME, "red"), (BLUE_HOME, "blue")):
        for l in home:
            facts.add(("accepts", (l, colour)))
            facts.add(("xnbr", (l, VOID, VOID)))
            facts.add(("ynbr", (l, VOID, VOID)))
    return facts


def make_initial_state():
    """All red blocks on the red home slots, all blue blocks on the blue home slots, grid empty."""
    placement = dict(zip(RED_BLOCKS, RED_HOME))
    placement.update(zip(BLUE_BLOCKS, BLUE_HOME))
    return placement


def state_from_placement(placement):
    """Builds the full symbolic state from a {block: location} dictionary."""
    state = static_facts()
    occupied = set(placement.values())
    for b, l in placement.items():
        state.add(("colored", (l, block_color(b))))
    for l in GRID_CELLS + RED_HOME + BLUE_HOME:
        if l not in occupied:
            state.add(("empty", (l,)))
    return state


def block_color(b):
    return "red" if b in RED_BLOCKS else "blue"


def apply_move_to_placement(placement, src, dst):
    """Updates the {block: location} map for a move and returns the block that was moved."""
    block = next(b for b, l in placement.items() if l == src)
    placement[block] = dst
    return block


class FastPlanner(sp.Planner):
    """Same A* as sp.Planner, but finds applicable actions with a per-predicate index of the
    state instead of scanning every fact for every precondition (same results, much faster)."""

    def applicable_actions(self, state):
        index = {}
        for name, args in sorted(state):   # sorted -> same plan on every run
            index.setdefault(name, []).append(args)
        out = []
        for schema in self.actions:
            pre = schema.preconds
            def backtrack(i, theta):
                if i == len(pre):
                    out.append(schema.ground(theta))
                    return
                name, pargs = pre[i]
                for args in index.get(name, ()):
                    if len(args) != len(pargs):
                        continue
                    new = dict(theta)
                    ok = True
                    for pv, sv in zip(pargs, args):
                        if sp.is_var(pv):
                            if new.get(pv, sv) != sv:
                                ok = False
                                break
                            new[pv] = sv
                        elif pv != sv:
                            ok = False
                            break
                    if ok:
                        backtrack(i + 1, new)
            backtrack(0, {})
        # the four axis variants can produce the same move; keep them all (A* dedupes states)
        return out


# Patterns from Figure 2, written row by row (row 0 = far from the robot).
# B = blue, R = red, . = empty cell
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


def home_goal():
    """Every block back on a home slot of its own colour (used for the optional reset)."""
    goal = {("colored", (l, "red")) for l in RED_HOME}
    goal |= {("colored", (l, "blue")) for l in BLUE_HOME}
    return goal


def goal_from_pattern(pattern):
    goal = set()
    for r, row in enumerate(pattern):
        for c, ch in enumerate(row):
            l = cell(r, c)
            if ch == "B":
                goal.add(("colored", (l, "blue")))
            elif ch == "R":
                goal.add(("colored", (l, "red")))
            else:
                goal.add(("empty", (l,)))
    return goal


def pattern_is_buildable(pattern):
    """Fast necessary check: can the target layout be built in SOME order?
    Run the build backwards: from the finished layout, keep removing any block that could be
    grasped (both neighbours along x, or both along y, empty). If every block can be removed,
    reversing that order builds the pattern. If we get stuck, no placement order exists, so
    the pattern is infeasible from any start state and the expensive search can be skipped."""
    occ = {(r, c) for r, row in enumerate(pattern) for c, ch in enumerate(row) if ch != "."}

    def free(r, c):
        return (r, c) not in occ

    progress = True
    while occ and progress:
        progress = False
        for (r, c) in sorted(occ):
            if (free(r - 1, c) and free(r + 1, c)) or (free(r, c - 1) and free(r, c + 1)):
                occ.discard((r, c))
                progress = True
                break
    return not occ


def parse_move(action):
    """move-xy(from, to, color) -> (from, to, color, pick_axis, place_axis)"""
    axes = action.name.split("-")[1]
    src, dst, color = action.args
    return src, dst, color, axes[0], axes[1]


if __name__ == "__main__":
    import time
    planner = FastPlanner(make_Grid_domain())
    placement = make_initial_state()
    for name, pattern in PATTERNS.items():
        t0 = time.time()
        plan = planner.plan(state_from_placement(placement), goal_from_pattern(pattern))
        if plan is None:
            print(f"[{name}] no plan found -> skipping")
            continue
        print(f"[{name}] {len(plan)} moves ({time.time() - t0:.1f}s)")
        for i, a in enumerate(plan, 1):
            src, dst, color, pa, qa = parse_move(a)
            block = apply_move_to_placement(placement, src, dst)
            print(f"  {i:02d}. {a.label():32s} -> moves {block}")
