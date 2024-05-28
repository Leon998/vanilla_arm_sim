import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import time
import numpy as np
from NLinkArm import NLinkArm
from utils.angle import angle_mod
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from math import pi

# Simulation parameters
N_LINKS = 3
show_animation = True
link_lengths = [1] * N_LINKS
q_init = np.array([-0.5*pi, -0.2*pi, 0])
goal_pos = np.array([1.5, -1])
t_span = np.linspace(0, 5)
control_mode = 0
A0 = np.array([[1, 0, 0], [0, 25, 0], [0, 0, 50]])
A1 = np.array([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
A2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 5]])
As = [A0, A1, A2]


def main():
    arm = NLinkArm(link_lengths, q_init, goal_pos, show_animation)
    A = As[control_mode]
    Q, T = DS(link_lengths, q_init, goal_pos, A)
    print(Q.shape)
    print(Q)
    print(T)
    np.savetxt('n_joint_arm_to_point_control/trajectory/A_' + str(control_mode) + '.txt', Q)
    for i, q in enumerate(Q):
        arm.update_joints(q)
        time.sleep(1./240.)

def DS(link_lengths, q_init, goal_pos, A):
    def odefun(t, q):
        J, J_inv = get_jacobian(link_lengths, q)
        current_pos = forward_kinematics(link_lengths, q)
        errors, _ = distance_to_goal(current_pos, goal_pos)
        dq = np.dot(A, np.dot(J.T, errors))
        return dq
    # sol = odeint(odefun, y0=q_init, t=t_span, tfirst=True)
    sol = solve_ivp(odefun, t_span=(0, max(t_span)), y0=q_init).y.T
    T = solve_ivp(odefun, t_span=(0, max(t_span)), y0=q_init).t
    return sol, T

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


if __name__ == '__main__':
    main()