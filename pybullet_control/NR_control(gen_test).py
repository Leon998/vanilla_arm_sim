import pybullet as p
import time
import pybullet_data
from math import pi
import numpy as np
from Three_Joint_Robot import ROBOT

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
init_joint_angles = [0.52, 0., 0.]
robot = ROBOT("arm_imitate", init_joint_angles)
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)
kpt_wrist = ROBOT.keypoint(robot, robot.wrist_index)
kpt_elbow = ROBOT.keypoint(robot, robot.elbow_index)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0,
                                 cameraPitch=-80, cameraTargetPosition=[0.5,0,0.5])

# goal_pos = np.array([0.5, 0.4, 0.5])
goal_pos = np.array([0.2, 0.6, 0.5])
p.addUserDebugPoints([goal_pos], [[1, 0, 0]], 20)
time.sleep(1)
# A = np.array([[1, 0, 0], [0, -4, 0], [0, 0, 6]])
q = 0

num_demo = 1
num_iter = 100
dt = 0.01

np.random.seed(1)
for j in range(num_demo):
    robot.FK(init_joint_angles)
    q = robot.q
    A = np.array([[5, 0, 0], 
                  [0, -20 + 10*np.random.random_sample()-5, 0], 
                  [0, 0, 30 + 10*np.random.random_sample()-5]])
    print(A)
    for i in range(num_iter):
        p.stepSimulation()
        time.sleep(1./240.)
        ## NR
        error = robot.get_error(goal_pos, robot.ee_index)
        robot.get_jacobian()
        q += A.dot(np.matmul(robot.J.T, error))*dt
        robot.FK(q)
        # time.sleep(0.05)
        kpt_ee.draw_traj()
        kpt_ee.save_traj()
        kpt_wrist.draw_traj()
        kpt_wrist.save_traj()
        kpt_elbow.draw_traj()
        kpt_elbow.save_traj()
np.savetxt("pybullet_control/trajectory/ee_test.txt", kpt_ee.traj)
np.savetxt("pybullet_control/trajectory/wrist_test.txt", kpt_wrist.traj)
np.savetxt("pybullet_control/trajectory/elbow_test.txt", kpt_elbow.traj)
# 断开连接
p.disconnect()