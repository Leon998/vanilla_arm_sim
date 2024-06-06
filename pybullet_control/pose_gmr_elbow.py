import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
import pydmps
import pydmps.dmp_discrete
from scipy.spatial.transform import Rotation as R


def traj_GMR(traj, start_pos, num_demo, num_iter, dt):
    X = traj[:, :2]
    X = X.reshape((num_demo, num_iter, -1))
    print(X.shape)  # (num_demo, num_iter, 2)
    X_train = []
    # compute velocity
    for x in X:
        x_dot = (x[1:] - x[:-1]) / dt
        x = x[:-1]
        x_train = np.hstack((x, x_dot))
        X_train.append(x_train)
    X_train = np.array(X_train)
    X_train = X_train.reshape(-1, 4)
    print(X_train.shape)  #(num_demo*(num_iter-1), 4)

    # GMR
    random_state = np.random.RandomState(0)
    n_components = 3
    initial_means = kmeansplusplus_initialization(X_train, n_components, random_state)
    initial_covs = covariance_initialization(X_train, n_components)
    bgmm = BayesianGaussianMixture(
        n_components=n_components, max_iter=500,
        random_state=random_state).fit(X_train)
    gmm = GMM(n_components=n_components, priors=bgmm.weights_, means=bgmm.means_,
              covariances=bgmm.covariances_, random_state=random_state)

    sampled_path = []
    x = start_pos
    sampling_dt = dt
    for t in range(num_iter):
        sampled_path.append(x)
        cgmm = gmm.condition([0, 1], x)
        ## default alpha defines the confidence region (e.g., 0.7 -> 70 %)
        # x_dot = cgmm.sample_confidence_region(1, alpha=0.7)[0]
        ## mean sampling
        x_dot = cgmm.to_mvn().mean
        x = x + sampling_dt * x_dot
    sampled_path = np.array(sampled_path)
    return X_train, sampled_path

if __name__=="__main__":
    num_demo = 10
    num_iter = 50
    dt = 0.1
    
    ee = np.loadtxt("pybullet_control/trajectory/ee_demo.txt")[:, :3].reshape((-1, 3))
    ee_q = np.loadtxt("pybullet_control/trajectory/ee_demo.txt")[:, 3:].reshape((-1, 4))
    eb = np.loadtxt("pybullet_control/trajectory/elbow_demo.txt").reshape((-1, 3))
    eb_ee = []
    for i, q in enumerate(ee_q):
        R_we = R.from_quat(q).as_matrix()
        eb_ee.append(R_we.T.dot(eb[i] - ee[i]))
    eb_ee = np.array(eb_ee).reshape((-1, 3))
    print(eb_ee.shape)
    # plt.figure()
    # plt.title("TP elbow")
    # plt.scatter(eb_ee[:, 0], eb_ee[:, 1],  marker='.', alpha=0.8, label="elbow_ee")
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    # plt.xlim(-0.7, 0)
    # plt.ylim(0, 0.7) 
    # plt.legend()
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()


    ## ee DMP trajectory generation
    # dmp_ee = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10.0)
    # dmp_ee.imitate_path(y_des=ee[:num_iter, :2].T)
    # ee_imitate = []
    # # changing start position
    # dmp_ee.y = np.array([0.9, 0.6])
    # # changing end position
    # dmp_ee.goal = np.array([0.3, 0.8])
    # for t in range(dmp_ee.timesteps):
    #     y, _, _ = dmp_ee.step()
    #     ee_imitate.append(np.copy(y))
    #     # move the target slightly every time step
    # ee_imitate = np.array(ee_imitate)
    # ee_repro = ee_imitate[::2]
    
    ## ee test trajectory
    # 直接用测试轨迹来做reproduction，因为DMP还需要学方向
    ee_repro = np.loadtxt("pybullet_control/trajectory/ee_test.txt")[:, :3].reshape((-1, 3))
    ee_test_q = np.loadtxt("pybullet_control/trajectory/ee_test.txt")[:, 3:].reshape((-1, 4))
    eb_true = np.loadtxt("pybullet_control/trajectory/elbow_test.txt").reshape((-1, 3))

    

    # GMR
    train_eb_ee, sampled_eb_ee = traj_GMR(eb_ee, np.array([-0.6, 0.]), num_demo, num_iter, dt)
    sampled_eb_ee = np.hstack((sampled_eb_ee, ee_repro[:, 2].reshape(-1, 1)))

    eb_repro = []
    for i, q in enumerate(ee_test_q):
        R_we = R.from_quat(q).as_matrix()
        eb_repro.append(R_we.dot(sampled_eb_ee[i]) + ee_repro[i])
    eb_repro = np.array(eb_repro).reshape((-1, 3))
    print(eb_repro.shape)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("elbow GMR")
    plt.scatter(train_eb_ee[:, 0], train_eb_ee[:, 1],  marker='.', alpha=0.8, label="Demonstration")
    plt.plot(sampled_eb_ee[:, 0], sampled_eb_ee[:, 1], "orange", lw=4, label="Reproduction")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim(-0.7, 0)
    plt.ylim(0, 0.7) 
    plt.legend()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("ee-elbow-wrist repro")
    plt.scatter(ee[:, 0], ee[:, 1], c="r", marker='.', alpha=0.1, label="ee_demo")
    plt.scatter(eb[:, 0], eb[:, 1], c="g", marker='.', alpha=0.1, label="eb_demo")
    plt.plot(ee_repro[:, 0], ee_repro[:, 1], "r", lw=2, label="ee_repro")
    plt.plot(eb_repro[:, 0], eb_repro[:, 1], "orange", lw=5, label="eb_repro")
    plt.plot(eb_true[:, 0], eb_true[:, 1], "g", lw=5, label="eb_true")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim(-0.5, 1)
    plt.ylim(-0.5, 1) 
    plt.legend()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # # # save repro trajectory
    # # z = ee[:num_iter, 2].reshape((-1, 1))
    # # np.savetxt("pybullet_control/trajectory/ee_repro.txt", np.concatenate((ee_repro, z),axis=1)[::5])
    # # np.savetxt("pybullet_control/trajectory/elbow_repro.txt", np.concatenate((eb_repro, z),axis=1)[::5])