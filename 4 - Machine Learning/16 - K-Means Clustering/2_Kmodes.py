## KModes Implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from kneed import KneeLocator
import warnings

warnings.filterwarnings("ignore")

def finding_best_k(raw, enc, max_k=20):
    costs, sils, ks = [], [], list(range(1, max_k + 1))
    for k in ks:
        km = KModes(n_clusters=k, init="Cao", n_init=5, verbose=0, random_state=42)
        labels = km.fit_predict(raw)
        costs.append(km.cost_)
        sil = silhouette_score(enc, labels, metric="hamming") if k > 1 else np.nan
        sils.append(sil)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, costs, "b-x", label="Cost (dissimilarity)")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Cost", color="b")
    ax2 = ax1.twinx()
    ax2.plot(ks, sils, "r-o", label="Silhouette")
    ax2.set_ylabel("Silhouette (Hamming)", color="r")
    plt.title("K‑modes: cost + silhouette (Hamming) vs k")
    fig.tight_layout()
    plt.show()
    
    # Finding the best K automatically (elbow method)
    kl = KneeLocator(ks, costs, curve="convex", direction="decreasing")
    elbow = kl.elbow or np.nanargmax(sils) + 1
    print(f"Elbow method suggests k = {elbow}")
    return elbow


def kmodes_clustering(k, raw, enc, df, label_encs):
    km = KModes(n_clusters=k, init="Cao", n_init=5, verbose=1, random_state=42)
    labels = km.fit_predict(raw)
    cost = km.cost_
    sil = silhouette_score(enc, labels, metric="hamming") if k > 1 else np.nan

    modes = pd.DataFrame(km.cluster_centroids_, columns=df.columns)
    for col, le in label_encs.items():
        modes[col] = modes[col].round().astype(int).map(lambda code: le.classes_[code])

    print(f"\nK‑modes clustering (k={k})")
    print(f"  Total cost      : {cost:.0f}")
    print(f"  Hamming silhouette: {sil:.3f}")
    print("Cluster modes per feature (centroids):")
    print(modes.to_string(index=False))
    print("\nCluster sizes:")
    print(pd.value_counts(labels).sort_index().rename("count"))

    model_labels = labels
    df["cluster"] = model_labels
    return km, model_labels, modes


if __name__ == "__main__":
    # 1. Load the car evaluation dataset from OpenML
    car = fetch_openml(name="car", version=2, as_frame=True)  # public, fully categorical :contentReference[oaicite:1]{index=1}
    df = car.frame.copy()

    print(f"Loaded 'car' dataset with shape {df.shape}")
    print("Column types:")
    print(df.dtypes)

    # 2. Label-encode each column independently
    label_encoders = {}
    df_enc = pd.DataFrame()
    for col in df.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    raw_array = df_enc.values          # integer codes for raw categorical data
    enc_array = df_enc.values          # same data for silhouette (metric='hamming')

    # 3. Find optimal k
    best_k = finding_best_k(raw_array, enc_array, max_k=10)

    # 4. Fit final K‑modes and print summary
    model, labels, modes = kmodes_clustering(best_k, raw_array, enc_array, df, label_encoders)

    # 5. Cluster counts
    print("\nCluster label counts in the original DataFrame:")
    print(df["cluster"].value_counts())

    # Optional: save output
    # df.to_csv("car_with_clusters.csv", index=False)