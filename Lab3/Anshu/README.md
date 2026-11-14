"""
Robot Autonomy - Lab 3: Task planning on the Franka Panda.

Pipeline
  1. Symbolic planning (BlockGridDomain + SymPlanner A*) finds a sequence of colour moves
     for each pattern, starting from wherever the blocks currently are. Infeasible patterns
     (no plan) are skipped.
  2. Each move becomes a pick-and-place: Cartesian waypoints (approach, grasp, lift, transit,
     place, release, retreat) with the gripper rotated so the fingers close along the
     axis the planner chose (the axis whose neighbouring cells are empty).
  3. Damped-least-squares IK (using the MuJoCo Panda model) turns those poses into
     7-DoF joint waypoints.
  4. The waypoints are executed in MuJoCo with the Lab 2 PD + gravity-compensation
     controller, and saved to disk so the same joint trajectory can be sent to the real robot.

Run
  python Lab3.py                 # plan, simulate with the viewer, save waypoints
  python Lab3.py --headless      # same without the viewer (prints block placement errors)
  python Lab3.py --reset         # return all blocks home between patterns
  python Lab3.py --plan-only     # only print the symbolic plans
  python Lab3.py --headless --record sim.mp4   # save a video of the simulation (needs ffmpeg)

Waypoint file format (one row per waypoint): [q1 ... q7, gripper]
  The arm moves to q (min-jerk); THEN the gripper is commanded to the row's gripper value
  (GRIPPER_OPEN / GRIPPER_CLOSED). If the gripper value changed, the arm holds still while
  the gripper moves.
"""
