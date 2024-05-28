import pybullet as p
import time
import pybullet_data
from math import pi
import numpy as np
from scipy.integrate import solve_ivp, odeint


def getJointStates(robot_id):
   joint_states = p.getJointStates(robot_id, range(p.getNumJoints(robot_id)))
   joint_positions = [state[0] for state in joint_states]
   joint_velocities = [state[1] for state in joint_states]
   joint_torques = [state[3] for state in joint_states]
   return joint_positions, joint_velocities, joint_torques

def error_distance(robot_id, goal_pos, index):
    current_pos = p.getLinkState(robot_id, index)[0]
    error = goal_pos - current_pos
    return error

def FK(robot_id, joints_indexes, q):
    for i in range(len(joints_indexes)):
        p.resetJointState(bodyUniqueId=robot_id,
                          jointIndex=i,
                          targetValue=q[i],
                          targetVelocity=0)

def DS(robot_id, joints_indexes, goal_pos, index, A, t_span=np.linspace(0, 5)):
    q_init, _, _ = getJointStates(robot_id)
    def odefun(t, q):
        FK(robot_id, joints_indexes, q)
        q_tmp, _, _ = getJointStates(robot_id)
        loc_p = p.getLinkState(robot_id, index)[2]
        zero_vec = [0.0] * p.getNumJoints(robot_id)
        J, jac_r = p.calculateJacobian(robot_id, index, loc_p, q_tmp, zero_vec, zero_vec)
        error = error_distance(robot_id, goal_pos, index)
        dq = np.dot(A, np.dot(np.array(J).T, error))
        return dq
    result = solve_ivp(odefun, t_span=(0, max(t_span)), y0=q_init)
    sol = result.y.T
    T = result.t
    return sol, T


# 连接物理引擎
physicsCilent = p.connect(p.GUI)

# 渲染逻辑
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# 设置环境重力加速度
p.setGravity(0, 0, 0)
# 加载URDF模型，此处是加载蓝白相间的陆地
planeId = p.loadURDF("plane.urdf")
# 加载机器人，并设置加载的机器人的位姿
startPos = [0, 0, 0.5]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("vanilla_control/3joint_arm/urdf/3joint_arm.urdf", 
                      startPos, startOrientation, useFixedBase=1)

joints_indexes = [i for i in range(p.getNumJoints(robot_id)) 
                  if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0,
                                 cameraPitch=-80, cameraTargetPosition=[0.5,0,0.5])

E_index = joints_indexes[-1]
zero_vec = [0.0] * p.getNumJoints(robot_id)
goal_pos = np.array([0.4, 0.6, 0.5])
p.addUserDebugPoints([goal_pos], [[1, 0, 0]], 20)
time.sleep(1)
A = np.array([[1, 0, 0], [0, -4, 0], [0, 0, 7]])
q_init, _, _ = getJointStates(robot_id)


t_span=np.linspace(0, 10)
Q, T = DS(robot_id, joints_indexes, goal_pos, E_index, A, t_span)
print(Q, T)


FK(robot_id, joints_indexes, q_init)
for (q, t) in zip(Q, T):
    p.stepSimulation()
    time.sleep(1./240.)
    FK(robot_id, joints_indexes, q)
