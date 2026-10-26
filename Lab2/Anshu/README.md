# Lab 2: Pick and Place with Franka Panda (MuJoCo)

## Setup
```bash
cd Lab2/Anshu
python3 -m venv venv && source venv/bin/activate
pip install mujoco numpy
```

## Run
```bash
python3 lab2.py
```

Run it from inside `Lab2/Anshu`.

## Files
- `lab2.py`: simulator, IK, RRT-Connect planner, pick/place, task
- `scene.py`: builds the table, shelves and block
- `RobotUtil.py`: course helpers
- `franka_emika_panda/`: robot model
