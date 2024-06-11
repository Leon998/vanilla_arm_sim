import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
import pydmps
import pydmps.dmp_discrete


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
        x_dot = cgmm.sample_confidence_region(1, alpha=0.7)[0]
        ## mean sampling
        # x_dot = cgmm.to_mvn().mean
        x = x + sampling_dt * x_dot
    sampled_path = np.array(sampled_path)
    return X_train, sampled_path

if __name__=="__main__":
    num_demo = 10
    num_iter = 50
    dt = 0.1
    
    ee = np.loadtxt("pybullet_control/trajectory/ee_demo.txt")[:, :3].reshape((-1, 3))
    eb = np.loadtxt("pybullet_control/trajectory/elbow_demo.txt").reshape((-1, 3))
    wr = np.loadtxt("pybullet_control/trajectory/wrist_demo.txt").reshape((-1, 3))
    eb_ee = eb - ee
    wr_ee = wr - ee

    # ee DMP trajectory generation
    dmp_ee = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10.0)
    dmp_ee.imitate_path(y_des=ee[:num_iter, :2].T)
    ee_imitate = []
    # changing start position
    dmp_ee.y = np.array([0.9, 0.6])
    # changing end position
    dmp_ee.goal = np.array([0.3, 0.8])
    for t in range(dmp_ee.timesteps):
        y, _, _ = dmp_ee.step()
        ee_imitate.append(np.copy(y))
        # move the target slightly every time step
    ee_imitate = np.array(ee_imitate)
    ee_repro = ee_imitate[::2]
    
    # ee_repro = ee[3*num_iter:4*num_iter, :2]
    

    # GMR
    train_eb_ee, sampled_eb_ee = traj_GMR(eb_ee, np.array([-0.52, -0.3]), num_demo, num_iter, dt)
    train_wr_ee, sampled_wr_ee = traj_GMR(wr_ee, np.array([-0.26, -0.15]), num_demo, num_iter, dt)

    eb_repro = sampled_eb_ee + ee_repro
    wr_repro = sampled_wr_ee + ee_repro
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("elbow GMR")
    plt.scatter(train_eb_ee[:, 0], train_eb_ee[:, 1],  marker='.', alpha=0.8, label="Demonstration")
    plt.plot(sampled_eb_ee[:, 0], sampled_eb_ee[:, 1], "orange", lw=4, label="Reproduction")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("wrist GMR")
    plt.scatter(train_wr_ee[:, 0], train_wr_ee[:, 1],  marker='.', alpha=0.8, label="Demonstration")
    plt.plot(sampled_wr_ee[:, 0], sampled_wr_ee[:, 1], "orange", lw=4, label="Reproduction")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("ee-elbow-wrist repro")
    plt.scatter(ee[:, 0], ee[:, 1], c="r", marker='.', alpha=0.1, label="ee_demo")
    plt.scatter(eb[:, 0], eb[:, 1], c="g", marker='.', alpha=0.1, label="eb_demo")
    plt.scatter(wr[:, 0], wr[:, 1], c="b", marker='.', alpha=0.1, label="wr_demo")
    plt.plot(ee_repro[:, 0], ee_repro[:, 1], "r", lw=2, label="ee_repro")
    plt.plot(eb_repro[:, 0], eb_repro[:, 1], "g", lw=2, label="eb_repro")
    plt.plot(wr_repro[:, 0], wr_repro[:, 1], "b", lw=2, label="wr_repro")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # save repro trajectory
    z = ee[:num_iter, 2].reshape((-1, 1))
    np.savetxt("pybullet_control/trajectory/ee_repro.txt", np.concatenate((ee_repro, z),axis=1)[::5])
    np.savetxt("pybullet_control/trajectory/elbow_repro.txt", np.concatenate((eb_repro, z),axis=1)[::5])
    np.savetxt("pybullet_control/trajectory/wrist_repro.txt", np.concatenate((wr_repro, z),axis=1)[::5])