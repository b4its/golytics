# randomForest/rekomendasi_predictor.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util
import torch
import random
import os


# === Load data prediksi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_prediksi = os.path.join(BASE_DIR, "../generate/dataset/dtPrediksi.csv")
df = pd.read_csv(path_prediksi)

# === Buat label rekomendasi dari aturan logika
def get_label(row):
    if row['status'] == 0 and row['jamOperasional'] > 10 and row['kerugian'] > 100000:
        return "status_jam_kerugian"
    elif row['status'] == 0 and row['keuntungan'] > 5000000:
        return "status_untung_tapi_tak_sehat"
    elif row['status'] == 0:
        return "status_tak_sehat"
    elif row['jamOperasional'] > 10:
        return "jam_berlebihan"
    elif row['kerugian'] > 100000:
        return "kerugian_tinggi"
    else:
        return "kinerja_baik"

df['label'] = df.apply(get_label, axis=1)

# === Fitur dan label
fitur = ['status', 'jamOperasional', 'kerugian', 'keuntungan']
X = df[fitur]
y = df['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y_encoded)

# === Kalimat rekomendasi untuk setiap label
rekomendasi_variatif = {
    "status_jam_kerugian": [
        "Jam operasional terlalu tinggi dan kondisi bisnis tidak sehat. Evaluasi efisiensi manajemen Anda.",
        "Bisnis berjalan tidak sehat dengan jam kerja berlebih serta kerugian signifikan. Segera tinjau ulang strategi operasional.",
    ],
    "status_untung_tapi_tak_sehat": [
        "Meskipun laba tinggi, status operasional kurang baik. Periksa kualitas SDM dan operasional.",
        "Keuntungan besar tapi ada isu kesehatan bisnis. Mungkin ada masalah di tingkat manajemen atau beban kerja.",
    ],
    "status_tak_sehat": [
        "Kondisi bisnis tidak sehat. Evaluasi cara kerja dan efisiensi operasional.",
        "Status operasional buruk. Perlu perbaikan dalam struktur tim atau pengelolaan waktu.",
    ],
    "jam_berlebihan": [
        "⏱Jam kerja melebihi batas ideal. Kurangi jam operasional untuk meningkatkan efisiensi.",
        "Durasi kerja terlalu panjang. Evaluasi produktivitas harian.",
    ],
    "kerugian_tinggi": [
        "Kerugian tinggi terdeteksi. Perlu efisiensi dalam pengeluaran.",
        "Tingkat kerugian tinggi. Tinjau ulang pos pengeluaran dan strategi pendapatan.",
    ],
    "kinerja_baik": [
        "Bisnis berjalan dengan baik. Pertahankan kinerja dan lakukan evaluasi rutin.",
        "Kinerja bisnis optimal. Teruskan strategi yang ada dan pantau tren ke depan.",
    ]
}

# === Load SBERT
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode semua kalimat rekomendasi
all_texts, label_map = [], []
for label, kalimat_list in rekomendasi_variatif.items():
    for text in kalimat_list:
        all_texts.append(text)
        label_map.append(label)
template_embeddings = sbert_model.encode(all_texts, convert_to_tensor=True)

# === Fungsi rekomendasi terbaik
def get_sbert_rekomendasi(label):
    mask = [i for i, l in enumerate(label_map) if l == label]
    if not mask:
        return "Tidak ada rekomendasi untuk label ini."
    subset_embeddings = template_embeddings[mask]
    subset_texts = [all_texts[i] for i in mask]
    query_text = random.choice(subset_texts)
    query_embedding = sbert_model.encode(query_text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, subset_embeddings)[0]
    best_idx = scores.argmax().item()
    return subset_texts[best_idx]

# === Prediksi dan simpan hasil
prediksi = rf.predict(X)
pred_label = le.inverse_transform(prediksi)

output_data = []

print("\n� Rekomendasi Otomatis Berbasis Random Forest + SBERT:\n")

for i, row in df.iterrows():
    label = pred_label[i]
    rekom = get_sbert_rekomendasi(label)
    output_data.append({
        "tanggal": row["tanggal"],
        "keputusan": label,
        "kata": rekom
    })

    print(f"� Tanggal: {row['tanggal']}")
    print(f" - {rekom}\n")

# Simpan ke CSV
df_output = pd.DataFrame(output_data).sort_values(by="tanggal")

output_path = os.path.join(BASE_DIR, "dataset/keputusanKata.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_output.to_csv(output_path, index=False)
print("Hasil rekomendasi telah disimpan ke dataset/keputusanKata.csv")
