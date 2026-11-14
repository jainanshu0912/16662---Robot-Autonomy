# Robot Autonomy – Lab 3: Task Planning

The robot rearranges 4 red and 4 blue blocks into the four patterns from the handout (pickaxe, axe, box, bow) on a tight 3×3 grid. It uses the HW3 symbolic planner (A*) for task planning, and IK-based pick-and-place for execution. Everything is tested in MuJoCo, and the same joint trajectories are saved for the real robot.

## Files

| File | What it does |
|---|---|
| `SymPlanner.py` | HW3 planner: STRIPS action schemas + A* search (unchanged API) |
| `BlockGridDomain.py` | Lab 3 domain: grid/home locations, the finger-collision rule, the four patterns, a fast feasibility check, and `FastPlanner` (same A*, indexed precondition matching) |
| `Lab3.py` | Full pipeline: plan → IK waypoints → collision check → simulate → save trajectories |
| `run_real_robot.py` | Sends the saved trajectories to the real Panda with frankapy (same calls as the Lab 2 real-robot script) |
| `RobotUtil.py`, `franka_emika_panda/` | From Lab 2 |

## How to run

```bash
python BlockGridDomain.py               # just the symbolic plans
python Lab3.py                          # plan + simulate in the viewer + save waypoints/
python Lab3.py --headless               # no viewer; prints block placement errors
python Lab3.py --reset                  # return blocks home between patterns
python Lab3.py --headless --record sim.mp4
python run_real_robot.py --dry-run      # print what would be sent to the robot
python run_real_robot.py                # real robot (hand on the e-stop!)
```

## How it works

**Symbolic domain.** There are 17 locations: 9 grid cells plus 4 home slots per colour. Blocks of the same colour are interchangeable, so the planner reasons about `colored(loc, colour)` and `empty(loc)` rather than block identities. This removes a huge amount of symmetry from the search, and the executor keeps track of which physical block is where.

**Finger-collision rule.** The grid spacing (6 cm centre to centre, 4 cm blocks) is tight enough that an open finger hits a block in the neighbouring cell. A top grasp, or a placement, with the fingers closing along x is therefore only allowed when both x-neighbours are empty, and likewise for y. Edge cells use a virtual, always-empty `void` neighbour. There are four `move-{pick axis}{place axis}` schemas, so the plan also says how to orient the gripper. The same rule stops the robot from dropping a block into the middle of a ring.

**Infeasible patterns are skipped.** Before searching, `pattern_is_buildable` runs the build backwards: it repeatedly removes any block in the target layout that has a free axis. If it gets stuck, no placement order exists, so the pattern is infeasible from any start. **Axe** is infeasible: its 2×2 block of B B / B R in the corner can never have its last block placed without a finger hitting a neighbour. The robot skips it, as the handout allows. Without this check, A* would only report failure after exhausting its 50,000 expansions (about 4 minutes).

**Motion.** Each move is turned into the following sequence:

1. Joint-space transit to above the source.
2. Vertical descent.
3. Close the gripper.
4. Lift.
5. Transit to above the destination.
6. Descend.
7. Open the gripper.
8. Lift.

IK is damped least squares on the MuJoCo Panda model. The gripper yaw is the planner's axis, choosing whichever of the two equivalent 180°-apart yaws is reachable and needs the least wrist motion. Every waypoint is collision-checked against the table and shelves. Transits are checked so the hand never dips below the approach height.

**Execution in sim.** Uses the Lab 2 PD + gravity-compensation controller with min-jerk interpolation, waiting for the arm to settle before each gripper action. In testing, all blocks end within about 4 mm of their target cells.

## Before running on the real robot

- **Measure and set** `GRID_CENTER`, `GRID_SPACING`, `HOME_X`, `RED_HOME_Y`, `BLUE_HOME_Y` and `TABLE_Z` at the top of `Lab3.py` to match where you put the blocks. These are in the robot base frame, in metres. Then re-run `python Lab3.py --headless` to regenerate `waypoints/`.
- Red goes on the robot's left (+y) and blue on its right (−y). Row 0 of each pattern is the row furthest from the robot.
- Do a dry run, then a first run with `SPEED_SCALE` increased and blocks removed.
- `run_real_robot.py` uses the same frankapy calls as `lab2_franka.py` (`reset_joints`, `goto_joints`, `open_gripper`, `close_gripper`). It moves to the Lab 2 start pose first and asks for Enter before each pattern.
