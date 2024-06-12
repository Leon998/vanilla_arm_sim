"""
DMP method requires target positions, which are un predictable in our case.
Here we assume some targets for test.
"""
print(__doc__)
import numpy as np
import os
import sys
sys.path.append(os.getcwd()+'/DMP')
import matplotlib.pyplot as plt
import pydmps
import pydmps.dmp_discrete


num_demo = 10
num_iter = 100
dt = 0.01


def kpt_dmp(kpt_demo, start_pos, end_pos):
    dmp_kpt = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10.0)
    dmp_kpt.imitate_path(y_des=kpt_demo)
    kpt_imitate = []
    dkpt_imitate = []
    # changing start position
    dmp_kpt.y = np.array(start_pos)
    # changing end position
    dmp_kpt.goal = np.array(end_pos)
    for t in range(dmp_kpt.timesteps):
        y, dy, _ = dmp_kpt.step(tau=1.0)  # tau is for making the system execute faster or slower
        kpt_imitate.append(np.copy(y))
        dkpt_imitate.append(np.copy(dy))
    kpt_imitate = np.array(kpt_imitate)
    dkpt_imitate = np.array(dkpt_imitate)
    return kpt_imitate, dkpt_imitate

def compute_velocity(x, dt):
    x_dot = (x.T[1:] - x.T[:-1]) / dt
    return x_dot.T

# ee
ee_demo = np.loadtxt("pybullet_control/trajectory/ee_demo.txt")[:num_iter, :2].T.reshape((2, -1))
dee_demo = compute_velocity(ee_demo, dt)
# wr_ee
wr_demo = np.loadtxt("pybullet_control/trajectory/wrist_demo.txt")[:num_iter, :2].T.reshape((2, -1))
dwr_demo = compute_velocity(wr_demo, dt)
wr_ee_demo = wr_demo - ee_demo
# eb_ee
eb_demo = np.loadtxt("pybullet_control/trajectory/elbow_demo.txt")[:num_iter, :2].T.reshape((2, -1))
deb_demo = compute_velocity(eb_demo, dt)
eb_ee_demo = eb_demo - ee_demo

# ee imitate
ee_imitate, dee_imitate = kpt_dmp(ee_demo, [1.05, 0.56], [0.28, 0.58])
## wr_ee imitate
wr_ee_imitate, dwr_ee_imitate = kpt_dmp(wr_ee_demo, [-0.25, -0.15], [0.13, -0.24])
wr_imitate = wr_ee_imitate + ee_imitate
dwr_imitate = dwr_ee_imitate + dee_imitate
## eb_ee imitate
eb_ee_imitate, deb_ee_imitate = kpt_dmp(eb_ee_demo, [-0.63, -0.29], [-0.24, -0.08])
eb_imitate = eb_ee_imitate + ee_imitate
deb_imitate = deb_ee_imitate + dee_imitate


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("DMP system")
plt.plot(ee_demo[0, :], ee_demo[1, :], "r--", lw=2, label="ee demo")
plt.plot(wr_demo[0, :], wr_demo[1, :], "g--", lw=2, label="wr demo")
plt.plot(eb_demo[0, :], eb_demo[1, :], "b--", lw=2, label="eb demo")
plt.plot(ee_imitate[:, 0], ee_imitate[:, 1], "r", lw=2, label="ee imitate")
plt.plot(wr_imitate[:, 0], wr_imitate[:, 1], "g", lw=2, label="wr imitate")
plt.plot(eb_imitate[:, 0], eb_imitate[:, 1], "b", lw=2, label="eb imitate")
plt.xlim(-0.1, 1.2)
plt.ylim(-0.1, 1.2)
ax = plt.gca()
ax.set_aspect('equal')
plt.legend()


t_demo = np.linspace(0, 1, len(dee_demo.T))
t_imitate = np.linspace(0, 1, len(ee_imitate))
plt.subplot(1, 3, 2)
plt.title("x velocity")
plt.plot(t_demo, dee_demo[0, :], "r--", lw=2, label="ee demo")
plt.plot(t_demo, dwr_demo[0, :], "g--", lw=2, label="wr demo")
plt.plot(t_demo, deb_demo[0, :], "b--", lw=2, label="eb demo")
plt.plot(t_imitate, dee_imitate[:, 0], "r", lw=2, label="ee imitate")
plt.plot(t_imitate, dwr_imitate[:, 0], "g", lw=2, label="wr imitate")
plt.plot(t_imitate, deb_imitate[:, 0], "b", lw=2, label="eb imitate")
plt.legend()

plt.subplot(1, 3, 3)
plt.title("y velocity")
plt.plot(t_demo, dee_demo[1, :], "r--", lw=2, label="ee demo")
plt.plot(t_demo, dwr_demo[1, :], "g--", lw=2, label="wr demo")
plt.plot(t_demo, deb_demo[1, :], "b--", lw=2, label="eb demo")
plt.plot(t_imitate, dee_imitate[:, 1], "r", lw=2, label="ee imitate")
plt.plot(t_imitate, dwr_imitate[:, 1], "g", lw=2, label="wr imitate")
plt.plot(t_imitate, deb_imitate[:, 1], "b", lw=2, label="eb imitate")
plt.legend()
plt.show()

# save repro trajectory
z = np.loadtxt("pybullet_control/trajectory/ee_demo.txt")[:len(ee_imitate), 2].reshape((-1, 1))
np.savetxt("pybullet_control/trajectory/ee_imitate.txt", np.concatenate((ee_imitate, z),axis=1)[::10])
np.savetxt("pybullet_control/trajectory/elbow_imitate.txt", np.concatenate((eb_imitate, z),axis=1)[::10])
np.savetxt("pybullet_control/trajectory/wrist_imitate.txt", np.concatenate((wr_imitate, z),axis=1)[::10])