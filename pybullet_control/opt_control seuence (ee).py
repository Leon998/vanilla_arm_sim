import pybullet as p
import time
import pybullet_data
from math import pi
import numpy as np
from scipy.optimize import minimize


def getJointStates(robot_id):
   joint_states = p.getJointStates(robot_id, range(p.getNumJoints(robot_id)))
   joint_positions = [state[0] for state in joint_states]
   joint_velocities = [state[1] for state in joint_states]
   joint_torques = [state[3] for state in joint_states]
   return joint_positions, joint_velocities, joint_torques

def error_distance(robot_id, goal_pos, index):
    current_pos = p.getLinkState(robot_id, index)[0]
    error = np.linalg.norm(goal_pos - current_pos)
    return error

def FK(robot_id, joints_indexes, q):
    for i in range(len(joints_indexes)):
        p.resetJointState(bodyUniqueId=robot_id,
                          jointIndex=i,
                          targetValue=q[i],
                          targetVelocity=0)

def opt_local(robot_id, joints_indexes, goal_pos, index):
    """
    分别对每个goal做关节优化，每一步只考虑局部最优，没有考虑全局最优
    """
    Q_star = []
    Error = []
    for g in goal_pos:
        q_init, _, _ = getJointStates(robot_id)
        def eqn(q):
            FK(robot_id, joints_indexes, q)
            error = error_distance(robot_id, g, index)
            return error
        q_star = minimize(eqn, q_init, method='BFGS')
        Q_star.append(q_star.x)
        Error.append(q_star.fun)
    Error = np.sum(np.array(Error))
    return Q_star, Error

def opt_global(robot_id, joints_indexes, goal_pos, index):
    """
    考虑全局最优
    """
    q_init = [0.,0.,0.,0.,0.,0.,0.,0.,0.]  # 长度为n×m（目标数×关节自由度数）
    def eqn(q):
        Error = []
        i = 0
        for g in goal_pos:
            FK(robot_id, joints_indexes, q[i:i+3])
            Error.append(error_distance(robot_id, g, index))
            i += 3
        # 以下几种误差的求法都可以
        # Error = np.linalg.norm(np.array(Error))
        Error = np.sum(np.array(Error))
        # Error = np.dot(np.array(Error).T, np.array(Error))  # QP问题
        return Error
    Q_star = minimize(eqn, q_init, method='BFGS')
    Error = Q_star.fun
    Q_star = np.array(Q_star.x).reshape([3,3])
    return Q_star, Error


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
robot_id = p.loadURDF("vanilla_control/arm_demo/urdf/arm_demo.urdf", 
                      startPos, startOrientation, useFixedBase=1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0,
                                 cameraPitch=-80, cameraTargetPosition=[0.5,0,0.5])


joints_indexes = [i for i in range(p.getNumJoints(robot_id)) 
                  if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
E_index = joints_indexes[-1]
goal_pos = np.array([[0.6, 0.2, 0.5],[0.5, 0.4, 0.5],[0.4, 0.6, 0.5]])
p.addUserDebugPoints(goal_pos, [[1, 0, 0],[1, 0, 0],[1, 0, 0]], 20)
time.sleep(1)

Q_star, Error = opt_global(robot_id, joints_indexes, goal_pos, E_index)
print("Error = ", Error)

while True:
    p.stepSimulation()
    time.sleep(1./240.)
    FK(robot_id, joints_indexes, [0.,0.,0.])
    time.sleep(0.5)
    for q_star in Q_star:
        print("q_star: ", q_star)
        FK(robot_id, joints_indexes, q_star)
        time.sleep(0.5)


    
# 断开连接
p.disconnect()