## Kmeans Implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

import warnings
warnings.filterwarnings('ignore')

##########################
### FUNCTIONS
##########################

## Finding the best K value
def finding_best_k(data, max_k=20):
    scores = []
    range_values = range(1, max_k)

    for k in range_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        scores.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range_values, scores, 'bx-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method For Optimal k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    kl = KneeLocator(range_values, scores, curve="convex", direction="decreasing")
    print(f"Optimal number of clusters: {kl.elbow}")
    return kl.elbow

## Applying Kmeans Clustering
def kmeans_clustering(optimal_k, data):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    silhouette = silhouette_score(data, labels)
    inertia = kmeans.inertia_

    print(f"\nKMeans Results (k={optimal_k})")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Inertia         : {inertia:.2f}")
    print(f"Cluster Centers : {centers.shape}")

    return kmeans, labels, centers


## Applying PCA
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return pca, reduced_data


## Plotting PCA Clusters
def plot_clusters(data_2d, labels, title="Cluster Visualization"):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=labels, palette='viridis', s=60)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


##############################
### MAIN WORKFLOW PIPELINE
##############################
# Load and scale data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Find optimal number of clusters
optimal_k_value = finding_best_k(scaled_data)

kmeans_model_before, labels_before, cluster_centers_before = kmeans_clustering(optimal_k = optimal_k_value, data = scaled_data)

# Reduce to 2D for visualization
pca, pca_data = apply_pca(scaled_data, n_components=2)

# Visualize clusters
plot_clusters(pca_data, labels_before, title=f"KMeans Clusters (k={optimal_k_value}) after PCA")

kmeans_after, labels_after, cluster_centers_after  = kmeans_clustering(optimal_k=optimal_k_value, data = pca_data)


## Adding cluster labels to dataframe
cluster_df = df.copy()
cluster_df['cluster_before_pca'] = labels_before
cluster_df['cluster_after_pca'] = labels_after

# cluster_df.to_csv("final_data_w_cluster.csv, index = False)