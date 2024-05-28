import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time
import numpy as np
from NLinkArm import NLinkArm
from utils.angle import angle_mod
from math import pi

# Simulation parameters
Kp = 1
dt = 0.1
N_LINKS = 3
N_ITERATIONS = 10000

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

show_animation = True

goal_pos = np.array([1.5, -1])

# control mode
control_mode = 1
A = np.array([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
I = np.identity(3)
dq_0 = np.array([3, 1, 2])
dt = 0.05

def main():  # pragma: no cover
    """
    Creates an arm using the NLinkArm class and uses its inverse kinematics
    to move it to the desired position.
    """
    link_lengths = [1] * N_LINKS
    joint_angles = np.array([-0.5*pi, -0.2*pi, 0])
    arm = NLinkArm(link_lengths, joint_angles, goal_pos, show_animation)
    i = 0
    while True:
        end_effector = arm.end_effector
        errors, distance = distance_to_goal(end_effector, goal_pos)
        if distance > 0.1:
            arm.joint_angles = NR(arm, link_lengths, arm.joint_angles, goal_pos, control_mode)
            i += 1
            print(i)
        arm.update_joints(arm.joint_angles)


def NR(arm, link_lengths, joint_angles, goal_pos, control_mode):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    current_pos = forward_kinematics(link_lengths, joint_angles)
    errors, distance = distance_to_goal(current_pos, goal_pos)
    J, J_inv = get_jacobian(link_lengths, joint_angles)
    if control_mode == 0:  # default
        joint_angles = joint_angles + np.matmul(J_inv, errors) * dt
    elif control_mode == 1: # joint augmented control
        joint_angles = joint_angles + A.dot(np.matmul(J.T, errors)) * dt
    elif control_mode == 2:  # null space control
        joint_angles = joint_angles + (np.matmul(J_inv, errors) + np.matmul(I - J_inv.dot(J), dq_0)) * dt
    # print("=====================")
    return joint_angles
    

def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T

def get_jacobian(link_lengths, joint_angles):
    J = np.zeros((2, N_LINKS))
    for i in range(N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, N_LINKS):
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

    return J, np.linalg.pinv(J)


def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return angle_mod(theta1 - theta2)


if __name__ == '__main__':
    main()