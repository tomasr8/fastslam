import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(0)

K = 4
means = np.array([[8.4, 8.2], [1.4, 1.6], [2.4, 5.4], [6.4, 2.4]])
covs = np.array([
    np.diag([0.1, 0.1]),
    np.diag([0.1, 0.1]),
    np.diag([0.1, 0.1]),
    np.diag([0.1, 0.1])
])

n_samples = 100

X = []
for mean, cov in zip(means, covs):
    x = np.random.multivariate_normal(mean, cov, n_samples)
    X += list(x)

X = np.array(X)
np.random.shuffle(X)


# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

print("Dataset shape:", X.shape)


weights = np.ones((K)) / K
means_est = means + np.random.normal(0, 0.1, size=(K, 2))
covs_est = covs.copy()
eps = 1e-8


for step in range(400):

    # visualize the learned clusters
    # if step % 1 == 0:
    #     plt.figure(figsize=(12, int(8)))
    #     plt.title("Iteration {}".format(step))
    #     axes = plt.gca()

    #     likelihood = []
    #     for j in range(k):
    #         likelihood.append(multivariate_normal.pdf(x=pos, mean=means[j], cov=cov[j]))
    #     likelihood = np.array(likelihood)
    #     predictions = np.argmax(likelihood, axis=0)

    #     for c in range(k):
    #         pred_ids = np.where(predictions == c)
    #         plt.scatter(pos[pred_ids[0], 0], pos[pred_ids[0], 1],
    #                     color=colors[c], alpha=0.2, edgecolors='none', marker='s')

    #     plt.scatter(X[..., 0], X[..., 1], facecolors='none', edgecolors='grey')

    #     for j in range(k):
    #         plt.scatter(means[j][0], means[j][1], color=colors[j])

    #     #plt.savefig("img_{0:02d}".format(step), bbox_inches='tight')
    #     plt.show()

    likelihood = []
    # Expectation step
    for j in range(K):
        likelihood.append(multivariate_normal.pdf(x=X, mean=means_est[j], cov=covs_est[j]))
    likelihood = np.array(likelihood)
    assert likelihood.shape == (K, len(X))

    b = []
    # Maximization step
    for j in range(K):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(K)], axis=0)+eps))

        # updage mean and variance
        means_est[j] = np.sum(b[j].reshape(len(X), 1) * X, axis=0) / (np.sum(b[j]+eps))
        covs_est[j] = np.dot((b[j].reshape(len(X), 1) * (X - means_est[j])).T, (X - means_est[j])) / (np.sum(b[j])+eps)

        # update the weights
        # weights[j] = np.mean(b[j])

        assert covs_est.shape == (K, X.shape[1], X.shape[1])
        assert means_est.shape == (K, X.shape[1])

    for j in range(K):
        weights[j] = np.mean(b[j])


print(means)
print(means_est)
print(covs_est)
print(weights)