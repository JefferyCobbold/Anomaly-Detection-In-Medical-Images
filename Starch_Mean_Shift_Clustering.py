import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import pandas as pd


df=pd.read_excel("/media/jeffery/B872-ED00/Project_Data/StarchData(2).xlsx")


# Generate sample data
#X, _ = df.iloc[:,:].values
X = df.select_dtypes(include=[np.number]).values
# Estimate the bandwidth (radius for clusters) for mean shift
bandwidth = estimate_bandwidth(X, quantile=0.2)

# Fit MeanShift clustering
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(X)

# Extract labels and cluster centers
labels = mean_shift.labels_
cluster_centers = mean_shift.cluster_centers_
n_clusters = len(np.unique(labels))

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=30, edgecolor='k', label='Data points')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', marker='X', label='Cluster centers')
plt.title(f'Mean Shift Clustering (Number of clusters: {n_clusters})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
