import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET

import RobotUtil as rt

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml" 
MODEL_XML = "franka_emika_panda/panda_torque_table_shelves.xml" 

# 7-DoF Panda joint waypoints in radians (examples)
WAYPOINTS = np.array([
    [0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
    [0.0, 0.65,  0.0, -2.0,  0.0,  2.65,  0.8, 0.04],
    [0.0, 0.65,  0.0, -2.0,  0.0,  2.65,  0.8, 0.0125],
    [1.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.0125],
    [1.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
    [0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04],
], dtype=float)

SEGMENT_DURATION = 2.0
HOLD_DURATION = 1.0

# Controller gains (per-joint)
KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)

# ---------------------------------------------------------

  
if __name__ == "__main__":
    #Load a model, add objects, and save a new model

    #Set random generator seed for reproducing results (useful for debugging)
    np.random.seed(13)

    #Generate scene and obstacle boxes for collison checking (points and axes)
    EndofTable=0.55+0.135+0.05
    modelTree = ET.parse(ROOT_MODEL_XML)

    BLOCKS=[
        ["TablePlane",[EndofTable-0.275,0.,-0.005],[0.275, 0.504, 0.0051]],
        ["LShelfDistal",[EndofTable-0.09-0.0225, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["LShelfProximal",[EndofTable-0.55-0.0225, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["LShelfBack",[EndofTable-0.55-0.0225-0.09, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["LShelfMid",[EndofTable-0.32, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["LShelfArch",[EndofTable-0.275-0.135+0.0225, 0.504-0.045-0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
        ["LShelfBottom",[EndofTable-0.275-0.135+0.0225, 0.504-0.09-0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
        ["LShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSupp2",[EndofTable-0.32-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, 0.504-0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfDistal",[EndofTable-0.09-0.0225, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["RShelfProximal",[EndofTable-0.55-0.0225, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["RShelfBack",[EndofTable-0.55-0.0225-0.09, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["RShelfMid",[EndofTable-0.32, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["RShelfArch",[EndofTable-0.275-0.135+0.0225, -0.504+0.045+0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfBottom",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
        ["RShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfMiddle",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.2],[0.2525, 0.135/2., 0.005]],
        ["RShelfMiddleSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.2],[0.315, 0.0225, 0.0225]],
        ["RShelfTop",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.4],[0.2525, 0.135/2., 0.005]],
        ["RShelfTopSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.4],[0.315, 0.0225, 0.0225]],
    ]

    for i in range(len(BLOCKS)):
        rt.add_free_block_to_model(tree=modelTree, name=BLOCKS[i][0], pos=BLOCKS[i][1], density= 20 , size=BLOCKS[i][2] , rgba=[0.2, 0.2, 0.9, 1],free=False)

    #Add a free block to manipulate
    rt.add_free_block_to_model(tree=modelTree, name="Block", pos=[EndofTable-0.145, 0.0, 0.05], density= 20 , size=[0.02, 0.02, 0.02] , rgba=[0.0, 0.9, 0.2, 1],free=True)

    modelTree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)
        
    ###### EXECUTE PLAN ######

    #Load the model
    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)
    arm_idx = [0,1,2,3,4,5,6];
    gripper_idx = 7

    # Initialize arm at first waypoint
    data.qpos[arm_idx] = WAYPOINTS[0][arm_idx]
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)

    # Map step duration to steps
    dt = model.opt.timestep
    segment_steps = max(1, int(SEGMENT_DURATION / dt))
    hold_steps =int(HOLD_DURATION/dt)

    #Launch Viewer
    v = viewer.launch_passive(model, data)
    v.cam.distance=3.0 #Shift camera 
    v.cam.azimuth += 90
    

    try:
        #Step through each waypoint
        for i in range(len(WAYPOINTS) - 1):
            q_start = WAYPOINTS[i][arm_idx].copy()
            q_goal  = WAYPOINTS[i + 1][arm_idx].copy()
            t = 0.0
            #Compute and control along a minjerk trajectory for each waypoint 
            for k in range(segment_steps+hold_steps):
                # Desired joint state 
                q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, SEGMENT_DURATION)

                # Current joint state
                q = data.qpos[arm_idx].copy()
                qd = data.qvel[arm_idx].copy()

                # Compute PD + gravity compensation torque
                data.ctrl[arm_idx] = 0.0
                tau = KP * (q_des - q) + KD * (qd_des - qd)
                data.ctrl[arm_idx] = tau+data.qfrc_bias[:7]
                data.ctrl[gripper_idx] = WAYPOINTS[i][-1]

                # Step sim
                mj.mj_step(model, data)
                v.sync()
                t += dt


    finally:
        if v is not None:
            v.close() #cleanup viewer at the end
