# improved-DBSCAN
DBSCAN improvement so that the algorithm works well with data with different densities

## Density-Based Clustering Algorithms
Density-Based Clustering refers to unsupervised learning methods that identify distinctive groups/clusters in the data, based on the idea that a cluster in data space is a contiguous region of high point density, separated from other such clusters by contiguous regions of low point density.

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a base algorithm for density-based clustering. It can discover clusters of different shapes and sizes from a large amount of data, which is containing noise and outliers.

The DBSCAN algorithm uses two parameters:

* ***minPts***: The minimum number of points (a threshold) clustered together for a region to be considered dense.
* ***eps (ε)***: A distance measure that will be used to locate the points in the neighborhood of any point.

These parameters can be understood if we explore two concepts called Density Reachability and Density Connectivity.

**Reachability** in terms of density establishes a point to be reachable from another if it lies within a particular distance (eps) from it.

**Connectivity**, on the other hand, involves a transitivity based chaining-approach to determine whether points are located in a particular cluster. For example, p and q points could be connected if p->r->s->t->q, where a->b means b is in the neighborhood of a.

There are three types of points after the DBSCAN clustering is complete:

* **Core:** This is a point that has at least m points within distance n from itself. 
* **Border:** This is a point that has at least one Core point at a distance n.
* **Noise:** This is a point that is neither a Core nor a Border. And it has less than m points within distance n from itself.

<img src="https://user-images.githubusercontent.com/55990659/203619157-9a36fb51-54b0-4fcb-be1c-c4b1bef57167.png" width="40%"/>

## Algorithmic steps for DBSCAN clustering

* The algorithm proceeds by arbitrarily picking up a point in the dataset (until all points have been visited).
* If there are at least ‘minPoint’ points within a radius of ‘ε’ to the point then we consider all these points to be part of the same cluster.
* The clusters are then expanded by recursively repeating the neighborhood calculation for each neighboring point.

for more information please visit the following link: <br />
“DBSCAN Clustering Algorithm in Machine Learning,” KDnuggets. https://www.kdnuggets.com/dbscan-clustering-algorithm-in-machine-learning.html

## Disadvantage with DBSCAN
The drawback of this method is that if the data have different densities in different parts, that is, for example, a part of the data space has a higher density and a part of the space has a lower density, this algorithm does not work well. The aim of this project is to improve the DBSCAN algorithm so that the algorithm works well with data with different densities. <br />
I generated 2-dimentional datasets with different densities and shapes and compared DBSCAN algorithm and its improved version on them.
The problem of this algorithm is that it uses fixed parameters in the implementation of the algorithm, which causes problems in clustering datasets that have different densities. To improve it, it is possible to implement an algorithm that uses different eps and minPts for each dataset or find the optimal parameters for each dataset.

### Example 1
<img src="https://user-images.githubusercontent.com/55990659/203623138-f60dfb19-70ec-413f-b428-553d1a3d2958.png" width="75%"/>

### Example 2
<img src="https://user-images.githubusercontent.com/55990659/203623062-d136a276-bd3d-48e6-8e39-1a013689ea19.png" width="75%"/>

