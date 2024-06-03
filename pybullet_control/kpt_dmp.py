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

# ee dmp
ee_demo = np.loadtxt("pybullet_control/trajectory/ee_traj.txt").T.reshape((3, -1))

# wr_ee dmp
wr_demo = np.loadtxt("pybullet_control/trajectory/wrist_traj.txt").T.reshape((3, -1))
wr_ee_demo = wr_demo - ee_demo

plt.figure(1, figsize=(6, 6))

# ee imitate
dmp_ee = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3) * 10.0)
dmp_ee.imitate_path(y_des=ee_demo)
ee_imitate = []
# changing start position
dmp_ee.y = np.array([0.9, 0.6, 0.5])
# changing end position
dmp_ee.goal = np.array([0.3, 0.8, 0.5])
for t in range(dmp_ee.timesteps):
    y, _, _ = dmp_ee.step()
    ee_imitate.append(np.copy(y))
    # move the target slightly every time step
ee_imitate = np.array(ee_imitate)

## wr imitate
# dmp_wr = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3) * 10.0)
# dmp_wr.imitate_path(y_des=wr_demo)
# wr_imitate = []
# # changing start position
# dmp_wr.y = np.array([0.75, 0.45, 0.5])
# # changing end position
# dmp_wr.goal = np.array([0.2, 0.5, 0.5])
# for t in range(dmp_wr.timesteps):
#     y, _, _ = dmp_wr.step()
#     wr_imitate.append(np.copy(y))
#     # move the target slightly every time step
# wr_imitate = np.array(wr_imitate)


## wr_ee imitate
dmp_wr_ee = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3) * 10.0)
dmp_wr_ee.imitate_path(y_des=wr_ee_demo)
wr_ee_imitate = []
# changing start position
dmp_wr_ee.y = np.array([-0.15, -0.15, 0.5])
# changing end position
dmp_wr_ee.goal = np.array([-0.1, -0.3, 0.5])
for t in range(dmp_wr_ee.timesteps):
    y, _, _ = dmp_wr_ee.step()
    wr_ee_imitate.append(np.copy(y))
    # move the target slightly every time step
wr_ee_imitate = np.array(wr_ee_imitate)
wr_imitate = wr_ee_imitate + ee_imitate



plt.plot(ee_demo[0, :], ee_demo[1, :], "r--", lw=2)
plt.plot(wr_demo[0, :], wr_demo[1, :], "r", lw=2)
plt.plot(ee_imitate[:, 0], ee_imitate[:, 1], "b--", lw=2)
plt.plot(wr_imitate[:, 0], wr_imitate[:, 1], "b", lw=2)
plt.title("DMP system")
plt.xlim(-0.1, 1.2)
plt.ylim(-0.1, 1.2)
ax = plt.gca()
ax.set_aspect('equal')
plt.legend(["ee demo", "wr demo", "ee imitate"])
plt.show()
