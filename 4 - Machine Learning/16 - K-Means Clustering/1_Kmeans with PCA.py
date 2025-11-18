## Kmeans Implementation
from cProfile import label
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

    # Visualizations
    n_features = data.shape[1]
    
    # If data is 2D, plot directly
    if n_features == 2:
        plt.figure(figsize=(10, 5))
        
        # Scatter plot with clusters
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='k')
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroids')
        plt.xlabel(f"Feature 1")
        plt.ylabel(f"Feature 2")
        plt.title(f"KMeans Clustering (k={optimal_k})")
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cluster size distribution
        plt.subplot(1, 2, 2)
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(unique, counts, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel("Cluster")
        plt.ylabel("Number of Samples")
        plt.title("Cluster Size Distribution")
        plt.xticks(unique)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    # # If data is >2D, use PCA for visualization
    # elif n_features > 2:
    #     pca_viz = PCA(n_components=2)
    #     data_2d = pca_viz.fit_transform(data)
    #     centers_2d = pca_viz.transform(centers)
        
    #     plt.figure(figsize=(15, 5))
        
    #     # Scatter plot with clusters (PCA)
    #     plt.subplot(1, 3, 1)
    #     scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='k')
    #     plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroids')
    #     plt.xlabel(f"PC1 ({pca_viz.explained_variance_ratio_[0]:.2%})")
    #     plt.ylabel(f"PC2 ({pca_viz.explained_variance_ratio_[1]:.2%})")
    #     plt.title(f"KMeans Clustering (k={optimal_k}) - PCA View")
    #     plt.colorbar(scatter, label='Cluster')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
        
    #     # Cluster size distribution
    #     plt.subplot(1, 3, 2)
    #     unique, counts = np.unique(labels, return_counts=True)
    #     colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(unique)))
    #     plt.bar(unique, counts, color=colors, edgecolor='black', alpha=0.7)
    #     plt.xlabel("Cluster")
    #     plt.ylabel("Number of Samples")
    #     plt.title("Cluster Size Distribution")
    #     plt.xticks(unique)
    #     plt.grid(True, alpha=0.3, axis='y')
        
    # Silhouette and Inertia metrics
    plt.subplot(1, 3, 3)
    metrics_labels = ['Silhouette\nScore', 'Inertia\n(normalized)']
    metrics_values = [silhouette, inertia / np.max([inertia, 1])]
    colors_metrics = ['#2ecc71', '#e74c3c']
    bars = plt.bar(metrics_labels, metrics_values, color = colors_metrics, alpha=0.7, edgecolor='black', linewidth=2)
    plt.ylabel("Score / Value")
    plt.title("Clustering Quality Metrics")
    plt.ylim([0, max(metrics_values) * 1.2])
    
    # Add value labels on bars
    # for bar, val in zip(bars, metrics_values):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    # plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

    return kmeans, labels, centers


## Applying PCA
def apply_pca(data, n_components=2, labels = None, title = None):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    
    # Plotting PCA Clusters
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='viridis', s=60)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return pca, reduced_data


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
pca, pca_data = apply_pca(scaled_data, n_components=2, labels=labels_before, 
                          title=f"KMeans Clusters (k={optimal_k_value}) after PCA")

kmeans_after, labels_after, cluster_centers_after  = kmeans_clustering(optimal_k=optimal_k_value, data = pca_data)


## Adding cluster labels to dataframe
cluster_df = df.copy()
cluster_df['cluster_before_pca'] = labels_before
cluster_df['cluster_after_pca'] = labels_after

# cluster_df.to_csv("final_data_w_cluster.csv, index = False)