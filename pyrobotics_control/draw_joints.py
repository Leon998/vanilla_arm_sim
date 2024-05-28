import numpy as np
import matplotlib.pyplot as plt


Q0 = np.loadtxt('n_joint_arm_to_point_control/trajectory/A_0.txt')
Q1 = np.loadtxt('n_joint_arm_to_point_control/trajectory/A_1.txt')
Q2 = np.loadtxt('n_joint_arm_to_point_control/trajectory/A_2.txt')

plt.subplot(2, 2, 1)
plt.title("q0")
plt.plot(np.linspace(0,5,len(Q0)), Q0[:,0])
plt.plot(np.linspace(0,5,len(Q1)), Q1[:,0])
plt.plot(np.linspace(0,5,len(Q2)), Q2[:,0])
plt.legend(['A0', 'A1', 'A2'])
plt.xlabel('t')
plt.ylabel('q')

plt.subplot(2, 2, 2)
plt.title("q1")
plt.plot(np.linspace(0,5,len(Q0)), Q0[:,1])
plt.plot(np.linspace(0,5,len(Q1)), Q1[:,1])
plt.plot(np.linspace(0,5,len(Q2)), Q2[:,1])
plt.legend(['A0', 'A1', 'A2'])
plt.xlabel('t')
plt.ylabel('q')

plt.subplot(2, 2, 3)
plt.title("q2")
plt.plot(np.linspace(0,5,len(Q0)), Q0[:,2])
plt.plot(np.linspace(0,5,len(Q1)), Q1[:,2])
plt.plot(np.linspace(0,5,len(Q2)), Q2[:,2])
plt.legend(['A0', 'A1', 'A2'])
plt.xlabel('t')
plt.ylabel('q')
plt.show()