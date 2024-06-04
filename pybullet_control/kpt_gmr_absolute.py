import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization


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
    
    ee = np.loadtxt("pybullet_control/trajectory/ee_traj.txt").reshape((-1, 3))
    eb = np.loadtxt("pybullet_control/trajectory/elbow_traj.txt").reshape((-1, 3))
    wr = np.loadtxt("pybullet_control/trajectory/wrist_traj.txt").reshape((-1, 3))

    # GMR
    train_eb, sampled_eb = traj_GMR(eb, np.array([0.4, 0.0]), num_demo, num_iter, dt)
    train_wr, sampled_wr = traj_GMR(wr, np.array([0.7, 0.0]), num_demo, num_iter, dt)

    ee_repo = ee[3*num_iter:4*num_iter, :2]
    eb_repo = sampled_eb
    wr_repo = sampled_wr
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("elbow GMR")
    plt.scatter(train_eb[:, 0], train_eb[:, 1],  marker='.', alpha=0.8, label="Demonstration")
    plt.plot(sampled_eb[:, 0], sampled_eb[:, 1], "orange", lw=4, label="Reproduction")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("wrist GMR")
    plt.scatter(train_wr[:, 0], train_wr[:, 1],  marker='.', alpha=0.8, label="Demonstration")
    plt.plot(sampled_wr[:, 0], sampled_wr[:, 1], "orange", lw=4, label="Reproduction")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("ee-elbow-wrist repo")
    plt.scatter(ee[:, 0], ee[:, 1], c="r", marker='.', alpha=0.1, label="ee_demo")
    plt.scatter(eb[:, 0], eb[:, 1], c="g", marker='.', alpha=0.1, label="eb_demo")
    plt.scatter(wr[:, 0], wr[:, 1], c="b", marker='.', alpha=0.1, label="wr_demo")
    plt.plot(ee_repo[:, 0], ee_repo[:, 1], "r", lw=2, label="ee_repo")
    plt.plot(eb_repo[:, 0], eb_repo[:, 1], "g", lw=2, label="eb_repo")
    plt.plot(wr_repo[:, 0], wr_repo[:, 1], "b", lw=2, label="wr_repo")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()

    plt.tight_layout()
    plt.show()