import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Buat folder output
os.makedirs("accuracy-test/text", exist_ok=True)

# Load dataset
df = pd.read_csv("randomForest/dataset/keputusanKata.csv")

# Ambil kolom teks dan label
sentences = df["kata"].tolist()
labels = df["keputusan"].tolist()

# Encode label ke numerik
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Embedding dengan Sentence-BERT
print("� Menghitung embedding dengan Sentence-BERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64)

# Normalisasi hasil embedding
embeddings = normalize(embeddings)

# =========================================
# === Training Pohon Keputusan ============
# =========================================
print("� Membuat pohon keputusan untuk label 'keputusan'...")
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(embeddings, labels_encoded)
preds = tree.predict(embeddings)

# Simpan visualisasi pohon keputusan
plt.figure(figsize=(24, 10))
plot_tree(
    tree,
    filled=True,
    feature_names=[f"f{i}" for i in range(embeddings.shape[1])],
    class_names=le.classes_
)
plt.title("Decision Tree untuk Klasifikasi Label 'keputusan'")
plt.savefig("accuracy-test/text/decision_tree_keputusan.png")
plt.show()
print("� Pohon keputusan disimpan ke 'accuracy-test/text/decision_tree_keputusan.png'")

# =========================================
# === Confusion Matrix & Classification ===
# =========================================
print("� Menampilkan confusion matrix dan laporan klasifikasi...")
cm = confusion_matrix(labels_encoded, preds)
report = classification_report(labels_encoded, preds, target_names=le.classes_)
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("accuracy-test/text/confusion_matrix_keputusan.png")
plt.show()
print("� Confusion matrix disimpan ke 'accuracy-test/text/confusion_matrix_keputusan.png'")
