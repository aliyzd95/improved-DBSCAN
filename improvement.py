import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics

d = 5
X = pd.read_csv(f'datasets/X{d}.csv')
X = np.array(X)
y = pd.read_csv(f'datasets/y{d}.csv')
y = np.array(y)
y = y.reshape((1000,))

eps_s = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
min_pts = [1, 5, 7, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50]
best_ad = 0
best_acc = 0
best_v = 0
best_s = 0
best_e = 0
best_m = 0
for e in eps_s:
    for m in min_pts:
        db = DBSCAN(eps=e, min_samples=m).fit(X)
        labels = db.labels_
        if d == 2 or d == 3:
            indices_one = labels == 1
            indices_zero = labels == 0
            labels[indices_one] = 0  # replacing 1s with 0s
            labels[indices_zero] = 1  # replacing 0s with 1s
        n_noise_ = list(labels).count(-1)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_noise_ < 200 and 1 < n_clusters_:
            # if metrics.accuracy_score(y, labels) > best_acc and metrics.v_measure_score(y, labels) > best_v:
            if metrics.adjusted_rand_score(y, labels) > best_ad:
                best_ad = metrics.adjusted_rand_score(y, labels)
                # best_s = metrics.silhouette_score(X, labels)
                # best_acc = metrics.accuracy_score(y, labels)
                # best_v = metrics.v_measure_score(y, labels)
                best_e = e
                best_m = m

print(f'best eps: {best_e}')
print(f'best min_samples: {best_m}')
db = DBSCAN(eps=best_e, min_samples=best_m).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
if d == 2 or d == 3:
    indices_one = labels == 1
    indices_zero = labels == 0
    labels[indices_one] = 0  # replacing 1s with 0s
    labels[indices_zero] = 1  # replacing 0s with 1s

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print(f'Accuracy: {metrics.accuracy_score(y, labels):0.3f}')
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(y, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(y, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col),
             markeredgecolor='k')

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col),
             markeredgecolor='k')

plt.title(f'improvement --- n_clusters: {n_clusters_}, n_noise: {n_noise_}, acc: {metrics.accuracy_score(y, labels)}')
plt.show()
