import pandas as pd, os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree

os.makedirs("accuracy-test/text", exist_ok=True)

# Load data
df = pd.read_csv("randomForest/dataset/keputusanKata.csv")

# Ambil data teks dan label
sentences = df["kata"].tolist()
labels = df["keputusan"].tolist()

# Encode label ke numerik
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
n_clusters = len(set(labels_encoded))

# SBERT Encoding
print("� Menghitung embedding dengan SBERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64)

# Normalisasi embedding
embeddings = normalize(embeddings)

# Clustering dengan KMeans
print(f"� Clustering dengan KMeans (n_clusters={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_preds = kmeans.fit_predict(embeddings)

# Evaluasi Clustering
ari = adjusted_rand_score(labels_encoded, cluster_preds)
sil_score = silhouette_score(embeddings, cluster_preds)

print("\n=== Evaluasi Clustering Kalimat ===")
print(f"• Adjusted Rand Index (ARI) : {ari:.4f}")
print(f"• Silhouette Score          : {sil_score:.4f}")

# Tambahkan hasil cluster ke DataFrame
df["cluster"] = cluster_preds

# Menentukan label dominan untuk tiap cluster
cluster_labels = (
    df.groupby("cluster")["keputusan"]
    .agg(lambda x: x.value_counts().idxmax())
)
df["cluster_label"] = df["cluster"].map(cluster_labels)

# ========================================
# === Visualisasi Clustering (PCA) =======
# ========================================
sampled_df = df.sample(n=1000, random_state=42)
sampled_idx = sampled_df.index
sampled_embeddings = embeddings[sampled_idx]
sampled_preds = cluster_preds[sampled_idx]

# Reduksi dimensi ke 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(sampled_embeddings)

# Plot clustering dengan label dominan
plt.figure(figsize=(10, 6))
palette = sns.color_palette("hsv", n_colors=n_clusters)

for i in range(n_clusters):
    idx = [j for j, label in enumerate(sampled_preds) if label == i]
    cluster_name = cluster_labels[i]
    plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"{cluster_name}", s=10)

plt.title("Visualisasi Clustering Sentence Bert + KMeans")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(markerscale=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy-test/text/hasil_clustering_keputusan.png")
print("Visualisasi disimpan ke 'hasil_clustering_keputusan.png'")


# =========================================
# === Visualisasi Pohon Keputusan =========
# =========================================
print("� Membuat pohon keputusan untuk label 'keputusan'...")
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(embeddings, labels_encoded)

plt.figure(figsize=(24, 10))
plot_tree(tree, filled=True, 
          feature_names=[f"f{i}" for i in range(embeddings.shape[1])], 
          class_names=le.classes_)
plt.title("Decision Tree untuk Klasifikasi Label 'keputusan'")
plt.savefig("accuracy-test/text/decision_tree_keputusan.png")
plt.show()
print("Pohon keputusan disimpan ke 'decision_tree_keputusan.png'")
