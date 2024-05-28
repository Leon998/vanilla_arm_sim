import pybullet as p
import time
import pybullet_data
from math import pi


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
robot_id = p.loadURDF("pybullet_control/model/arm_demo/urdf/arm_demo.urdf", 
                      startPos, startOrientation, useFixedBase=1)

joints_indexes = [i for i in range(p.getNumJoints(robot_id)) 
                  if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0,
                                 cameraPitch=-80, cameraTargetPosition=[0.5,0,0.5])

q = 0.01
hasPrevPose = 0
prevPose = 0

def draw_traj(robot_id, index, hasPrevPose, prevPose):
    ls = p.getLinkState(robot_id, index)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, ls[0], [1, 0, 0], 1, 15)
    return ls[0], 1

while True:
    p.stepSimulation()
    time.sleep(1./240.)
    q += 0.01
    p.resetJointState(bodyUniqueId=robot_id,
                      jointIndex=0,
                      targetValue=q,
                      targetVelocity=0)
    # keypoint tracking
    prevPose, hasPrevPose = draw_traj(robot_id, 0, hasPrevPose, prevPose)

# 断开连接
p.disconnect()