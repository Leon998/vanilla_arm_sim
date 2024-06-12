import pybullet as p
import time
import pybullet_data
from math import pi
import numpy as np
from scipy.optimize import minimize
from Three_Joint_Robot import ROBOT


def opt_kpt(robot, sample_len, ee_traj, wrist_traj, elbow_traj):
    """
    考虑全局最优
    """
    q_init = [0. for i in range(sample_len*3)]  # 长度为n×m（目标数×关节自由度数）
    def eqn(q):
        Error = []
        i, j, k= 0, 0, 0
        for g in ee_traj:
            robot.FK(q[i:i+3])
            Error.append(np.linalg.norm(robot.get_error(g, robot.ee_index)))
            i += 3
        for g in wrist_traj:
            robot.FK(q[j:j+3])
            Error.append(np.linalg.norm(robot.get_error(g, robot.wrist_index)))
            j += 3
        for g in elbow_traj:
            robot.FK(q[k:k+3])
            Error.append(np.linalg.norm(robot.get_error(g, robot.elbow_index)))
            k += 3
        # 以下几种误差的求法都可以，QP最好
        # Error = np.linalg.norm(np.array(Error))
        # Error = np.sum(np.array(Error))
        Error = np.dot(np.array(Error).T, np.array(Error))  # QP问题
        return Error
    Q_star = minimize(eqn, q_init, method='BFGS')
    Error = Q_star.fun
    Q_star = np.array(Q_star.x).reshape([sample_len,3])
    return Q_star, Error

def opt_ee(robot, sample_len, ee_traj):
    """
    考虑全局最优
    """
    q_init = [0. for i in range(sample_len*3)]  # 长度为n×m（目标数×关节自由度数）
    def eqn(q):
        Error = []
        i = 0
        for g in ee_traj:
            robot.FK(q[i:i+3])
            Error.append(np.linalg.norm(robot.get_error(g, robot.ee_index)))
            i += 3
        # 以下几种误差的求法都可以，QP最好
        # Error = np.linalg.norm(np.array(Error))
        # Error = np.sum(np.array(Error))
        Error = np.dot(np.array(Error).T, np.array(Error))  # QP问题
        return Error
    Q_star = minimize(eqn, q_init, method='BFGS')
    Error = Q_star.fun
    Q_star = np.array(Q_star.x).reshape([sample_len,3])
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
# 初始化机器人
robot = ROBOT("arm_imitate")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)
kpt_wrist = ROBOT.keypoint(robot, robot.wrist_index)
kpt_elbow = ROBOT.keypoint(robot, robot.elbow_index)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0,
                                 cameraPitch=-80, cameraTargetPosition=[0.5,0,0.5])


ee_imitate = np.loadtxt("pybullet_control/trajectory/ee_imitate.txt")
wrist_imitate = np.loadtxt("pybullet_control/trajectory/wrist_imitate.txt")
elbow_imitate = np.loadtxt("pybullet_control/trajectory/elbow_imitate.txt")
sample_len = len(ee_imitate)

p.addUserDebugPoints(ee_imitate, [([1, 0, 0]) for i in range(len(ee_imitate))], 5)
p.addUserDebugPoints(wrist_imitate, [([1, 0, 0]) for i in range(len(wrist_imitate))], 5)
p.addUserDebugPoints(elbow_imitate, [([1, 0, 0]) for i in range(len(elbow_imitate))], 5)
time.sleep(1)

Q_star, Error = opt_kpt(robot, sample_len, ee_imitate, wrist_imitate, elbow_imitate)
# Q_star, Error = opt_ee(robot, sample_len, ee_imitate)
print("Q_star = ", Q_star)
print("Error = ", Error)

while True:
    p.stepSimulation()
    time.sleep(1./240.)
    robot.FK([0.,0.,0.])
    time.sleep(0.5)
    for q_star in Q_star:
        print("q_star: ", q_star)
        robot.FK(q_star)
        time.sleep(0.25)
    
# 断开连接
p.disconnect()