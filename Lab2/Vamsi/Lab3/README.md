# Lab 3: Task Planning — running on the Franka

The robot rearranges 4 red and 4 blue blocks on a 3×3 grid into the patterns
pickaxe, axe, box and bow. The axe is impossible with top grasps (it needs a
filled 2×2 square, so the last block's fingers always hit a neighbour), and
the code skips it, as the handout asks.

**Status:** this works in simulation. It has not been run on the real robot
yet, so treat the first runs as tests and follow the staged order below.

## Safety

- Never run alone. One person keeps a hand on the emergency stop the whole time.
- The robot runs open loop: it assumes every block is exactly where the code
  says it is. It does not see the blocks.
- Do the raised, no-blocks run (step 2 below) before any run with blocks.

## What you need

Copy the whole `Lab2/Vamsi` folder, not just `Lab3/`. The Lab 3 scripts use
the Lab 2 files in the folder above them.

```
Lab2/Vamsi/
  MotionPlanner.py  TaskPlanner.py  SimpleScript.py  Franka.py  RobotUtil.py
  franka_emika_panda/
  Lab3/
    Lab3_Robot.py       <- run this on the robot
    Lab3_Sim.py         <- simulation
    Lab3_Shared.py      <- planning shared by both
    SymScript_Lab3.py   <- task planner + table measurements
    SymPlanner.py
```

Python packages on the robot PC: `numpy`, `matplotlib`, `frankapy`.
MuJoCo is only needed for the simulation.

## 1. Set the table measurements

Open `SymScript_Lab3.py` and find the block of constants above
`location_xyz`. All values are in metres, in the robot base frame
(x forward from the base, y to the robot's left, z up from the table).

| Constant | Meaning | Current value |
|---|---|---|
| `GRID_CENTER` | x, y of the centre grid cell | (0.46, 0.00) |
| `GRID_PITCH` | distance between neighbouring cell centres | 0.06 |
| `SLOT_X` | x of the four start slots on each side | 0.58, 0.50, 0.42, 0.34 |
| `SLOT_Y_RED` / `SLOT_Y_BLUE` | y of the red / blue start slots | +0.15 / −0.15 |
| `BLOCK_HALF` | half the block size | 0.02 (4 cm cubes) |

Mark these positions on the table (tape or a printed sheet) and check them
with a ruler. If you change `SLOT_Y_*` beyond about 0.155, the planner will
refuse some grasps because the hand gets too close to the shelves.

## 2. Lay out the blocks

Seen from the robot:

```
   left (+y)                         right (−y)
   red  @ x=0.58     [ ][ ][ ]       blue @ x=0.58
   red  @ x=0.50     [ ][ ][ ]       blue @ x=0.50
   red  @ x=0.42     [ ][ ][ ]       blue @ x=0.42
   red  @ x=0.34                     blue @ x=0.34
                      (robot)
```

Red blocks go in the four slots on the robot's left, blue on its right, and
the grid starts empty. Place each block squarely on its mark.

## 3. Start the robot

Bring the robot up the same way as for Lab 2 (unlock the joints, start the
frankapy control-PC connection, use the terminal where
`from frankapy import FrankaArm` works).

## 4. Run, in this order

```bash
cd Lab2/Vamsi/Lab3

# 1. No robot motion. Checks that planning works on this PC.
python Lab3_Robot.py --dry-run

# 2. NO BLOCKS on the table. Arm stays 5 cm high, gripper off,
#    waits for Enter before each action.
python Lab3_Robot.py --z-offset 0.05 --no-gripper --step --patterns pickaxe

# 3. Blocks in place. One pattern, Enter before each action.
python Lab3_Robot.py --step --patterns pickaxe

# 4. Full run. Record the video during this one.
python Lab3_Robot.py
```

A real run prints a reminder and waits for you to type `go`. The arm then
resets, moves to the home pose, opens the gripper and starts. The full run
is about 30 pick/place actions and takes roughly 9 minutes at the default
speed.

In step 2, check three things before moving on:

- the fingers come down centred over each slot and grid cell;
- the gripper turns 90° between `left_right` and `up_down` grasps, matching
  the printed plan;
- the arm stays clear of the shelves.

## Options

| Option | What it does |
|---|---|
| `--dry-run` | Plan only; print what would be sent. Add `--verbose` to see every command. |
| `--step` | Wait for Enter before every pick or place. Type `q` to stop. |
| `--patterns pickaxe box` | Run only the listed patterns. |
| `--reset` | Put all blocks back in the start slots between patterns (about twice as long). |
| `--z-offset 0.01` | Raise every grasp by this many metres. 0.01 gives the 3 cm grasp height used in Lab 2. |
| `--speed 0.3` | Scale on joint speed. Default 0.5; lower is slower. |
| `--no-gripper` | Never open or close the gripper. |
| `--yes` | Skip the `go` confirmation. |

## Reading the output

Each pattern prints its plan, for example:

```
01. pick(sR3, blue, left_right)
02. place(g01, blue, up_down)
```

- `sL0`–`sL3` are the red start slots, `sR0`–`sR3` the blue ones (0 is
  farthest from the robot).
- `g<row><col>` is a grid cell: row 0 is farthest from the robot, column 0 is
  on the robot's left.
- `left_right` means the fingers close along y; `up_down` means they close
  along x. The planner picks whichever direction has no block in the way.

`[axe] infeasible ... -> skipping` is expected.

## If something goes wrong

| Problem | What to do |
|---|---|
| `Lab 2 files ... not found; set LAB2_DIR` | `Lab3/` is not inside the Lab 2 folder. Move it back, or run with `LAB2_DIR=/path/to/Lab2/Vamsi python Lab3_Robot.py ...`. |
| `robot is ... rad away from the start of the planned path` | The arm is not where the plan expects (for example after an e-stop). Restart the script so it resets and re-plans. |
| `no grasp found for ...` | A slot or cell is out of reach or too close to the shelves with the current measurements. Recheck the constants in step 1. |
| Error mentioning `goto_joints` or `get_joints` | The installed frankapy may use different arguments. The calls are in `FrankaExecutor` and `main` in `Lab3_Robot.py`; `goto_joints(q, duration=...)` and `get_joints()` are the two that Lab 2's robot script did not use. |
| Fingers touch the table or grasp too low | Add `--z-offset 0.01`. |
| Fingers hit a neighbouring block | Stop. The grid on the table does not match the constants, or the grasp directions are swapped; rerun step 2 and compare with the printed plan. |
| Gripper jams or drops a block | Keep going for the video; the handout says this is not penalised. |

## Simulation (optional)

On a PC with MuJoCo:

```bash
python Lab3_Sim.py              # with viewer
python Lab3_Sim.py --headless   # faster, no viewer
```

This runs the same plan and checks every block position after every action.
