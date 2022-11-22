import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_dataset(d):
    scaler = StandardScaler()
    if d == 1:
        X, y = make_moons(n_samples=1000, noise=0.2)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('datasets/X1.csv', index=False)
        df[['label']].to_csv('datasets/y1.csv', index=False)
        plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 2:
        X, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=[1, 0.25], center_box=[-2.5, 2.5])
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('datasets/X2.csv', index=False)
        df[['label']].to_csv('datasets/y2.csv', index=False)
        plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 3:
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('datasets/X3.csv', index=False)
        df[['label']].to_csv('datasets/y3.csv', index=False)
        plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 4:
        X, y = make_circles(n_samples=1000, noise=0.15, factor=0.1)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('datasets/X4.csv', index=False)
        df[['label']].to_csv('datasets/y4.csv', index=False)
        plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 5:
        X, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=[0.5, 0.1], center_box=[-1, 1])
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X_aniso = np.dot(X, transformation)
        X_aniso = scaler.fit_transform(X_aniso)
        df = pd.DataFrame(dict(x=X_aniso[:, 0], y=X_aniso[:, 1], label=y))
        df[['x', 'y']].to_csv('datasets/X5.csv', index=False)
        df[['label']].to_csv('datasets/y5.csv', index=False)
        plt.scatter(X_aniso[:, 0], X_aniso[:, 1], marker='.')

    else:
        raise Exception('not true!')


make_dataset(5)
plt.show()
