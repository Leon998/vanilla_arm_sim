import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
from gmr import MVN, GMM, plot_error_ellipses


num_demo = 10
num_iter = 50
dt = 0.1

ee = np.loadtxt("pybullet_control/trajectory/ee_traj.txt").reshape((-1, 3))
eb = np.loadtxt("pybullet_control/trajectory/elbow_traj.txt").reshape((-1, 3))
wr = np.loadtxt("pybullet_control/trajectory/wrist_traj.txt").reshape((-1, 3))
eb_ee = eb - ee
wr_ee = wr - ee

X = wr_ee[:, :2]
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
print(X_train.shape, X_train[0])  #(num_demo*(num_iter-1), 4)
X_train = X_train[:, [0,2]]  #(num_demo*(num_iter-1), 2) 只看某个方向上的位置-速度关系
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
x = np.array([0.98])
sampling_dt = dt
plt.figure(figsize=(5, 5))
for t in range(num_iter):
    sampled_path.append(x)
    cgmm = gmm.condition([0], x)
    ## default alpha defines the confidence region (e.g., 0.7 -> 70 %)
    x_dot = cgmm.sample_confidence_region(1, alpha=0.7)[0]
    ## mean sampling
    # x_dot = cgmm.to_mvn().mean
    x = x + sampling_dt * x_dot
sampled_path = np.array(sampled_path)
print(sampled_path.shape)

plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.8, label="Demonstration")
# plt.plot(sampled_path[:, 0], sampled_path[:, 1], label="Reproduction")
plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"], factors=np.linspace(0.25, 2.0, 4))
plt.xlabel("$x$")
plt.ylabel("$x_dot$")
plt.legend()
plt.tight_layout()
plt.show()