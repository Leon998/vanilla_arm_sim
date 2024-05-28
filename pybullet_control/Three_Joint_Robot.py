import pybullet as p
import numpy as np


class ROBOT:
    def __init__(self, name):
        startPos = [0, 0, 0.5]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("pybullet_control/model/"+name+"/urdf/"+name+".urdf", startPos, startOrientation, useFixedBase=1)
        self.joints_indexes = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        self.elbow_index, self.wrist_index, self.ee_index = self.joints_indexes[0], self.joints_indexes[1], self.joints_indexes[2]
        self.q_init, self.dq_init, self.ddq_init = self.get_joints_states()

    def get_joints_states(self):
        joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques
    
    def get_error(self, goal_pos, index, ee=True):
        current_pos = p.getLinkState(self.robot_id, index)[0]
        error = goal_pos - current_pos
        return error
    
    def FK(self, q):
        for i in range(len(self.joints_indexes)):
            p.resetJointState(bodyUniqueId=self.robot_id,
                              jointIndex=i,
                              targetValue=q[i],
                              targetVelocity=0)
        self.q, self.dq, self.ddq = self.get_joints_states()

    def get_jacobian(self):
        self.q, self.dq, self.ddq = self.get_joints_states()
        zero_vec = [0.0] * p.getNumJoints(self.robot_id)
        jac_t, jac_r = p.calculateJacobian(self.robot_id, self.ee_index, p.getLinkState(self.robot_id, self.ee_index)[2], self.q, self.dq, zero_vec)
        self.J = np.array(jac_t)

    class keypoint:
        def __init__(self, robot, index, hasPrevPose=0, prevPose=0):
            self.robot = robot
            self.robot_id = robot.robot_id
            self.index = index
            self.hasPrevPose = hasPrevPose
            self.prevPose = prevPose
            self.traj = []
            pass

        def save_traj(self):
            ls = p.getLinkState(self.robot_id, self.index)
            self.traj.append(ls[0])
            
        def draw_traj(self):
            ls = p.getLinkState(self.robot_id, self.index)
            if (self.hasPrevPose):
                p.addUserDebugLine(self.prevPose, ls[0], [1, 0, 0], 3, 15)
            self.hasPrevPose = 1
            self.prevPose = ls[0]

        def reset_pose(self):
            self.hasPrevPose = 0
            self.prevPose = 0