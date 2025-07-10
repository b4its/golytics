import pandas as pd, random
import torch
import joblib
import os
from datetime import datetime
from models import StatusMLP, ProfitMLP
from sklearn.preprocessing import LabelEncoder
from arima.arimaPredict import forecast_arima
from randomForest.recommendationPredict import rekomendasi_variatif
# === Format Rupiah
def format_rupiah(angka):
    angka = max(0, angka)  # ubah jadi 0 kalau negatif
    return f"Rp {angka:,.0f}".replace(",", ".")


# === Load Data & Encoder
df_all = pd.read_csv("generate/dataset/dtPrediksi.csv")
le = LabelEncoder()
le.fit(df_all['jenisBisnis'])
known_labels = set(le.classes_)

# === Load Model Status
use_rf_model = False
use_status_model = False

if os.path.exists("randomForest/models/status_rf_model.pkl"):
    status_model = joblib.load("randomForest/models/status_rf_model.pkl")
    use_rf_model = True
    print("Menggunakan model Random Forest untuk prediksi status.")
elif os.path.exists("randomForest/models/status_predictor.pth"):
    status_model = StatusMLP(input_dim=3)
    status_model.load_state_dict(torch.load("randomForest/models/status_predictor.pth"))
    status_model.eval()
    status_model.double()
    status_scaler = joblib.load("randomForest/scalers/status_scaler.pkl")
    use_status_model = True
    print("Menggunakan model StatusMLP.")
else:
    print("Tidak ada model status ditemukan. Akan gunakan logika sederhana.")

# === Load Profit Model
profit_model = ProfitMLP(input_dim=5)
profit_model.load_state_dict(torch.load("mlp/models/profit_predictor.pth"))
profit_model.eval()
profit_model.double()
profit_scaler = joblib.load("mlp/scalers/profit_scaler.pkl")
profit_target_scaler = joblib.load("mlp/scalers/profit_target_scaler.pkl")

# === Load Dataset
path_pred = "generate/dataset/dtPrediksi.csv"
df = pd.read_csv(path_pred)

if df.empty:
    print("Tidak ada data prediksi tersedia.")
    exit()

# === Forecast ARIMA
forecast_path = "generate/dataset/hasilForecast_ARIMA.csv"
if not os.path.exists(forecast_path):
    print("Membuat prediksi tren laba bersih mingguan (ARIMA)...")
    forecast_arima(path_pred, forecast_path)
else:
    print("Forecast ARIMA sudah tersedia.")

# === Pilih Bisnis
print("\nForecast Mingguan (ARIMA):")
if os.path.exists(forecast_path):
    df_forecast = pd.read_csv(forecast_path)
    bisnis_unik = df['namaBisnis'].unique()

    for i, b in enumerate(bisnis_unik, 1):
        print(f"{i}. {b}")

    try:
        idx = int(input("\nPilih nomor bisnis untuk melihat forecast & hasil prediksi terakhir: "))
        nama_bisnis_pilih = bisnis_unik[idx - 1]
    except:
        print("Pilihan tidak valid.")
        exit()

    df_bisnis = df[df['namaBisnis'] == nama_bisnis_pilih]
    if df_bisnis.empty:
        print("Data bisnis tidak ditemukan.")
        exit()

    print(f"\nForecast untuk: {nama_bisnis_pilih}")
    print("\nTanggal       | Prediksi Laba Bersih")
    print("--------------------------------------")
    for row in df_forecast.itertuples():
        print(f"{row.tanggal} | {format_rupiah(row.prediksi_laba_bersih)}")
else:
    print("File forecast ARIMA tidak tersedia.")

# === Tampilkan Hasil Prediksi Terakhir Berdasarkan Bisnis
print(f"\nHasil Prediksi Terakhir untuk: {nama_bisnis_pilih}")
df_bisnis_last = df[df['namaBisnis'] == nama_bisnis_pilih]
df_new = df_bisnis_last.tail(5)

print("\nTanggal | Bisnis | Modal | Pemasukan | Pengeluaran | Jam | Status | Keuntungan | Kerugian | Laba Bersih")
for row in df_new.itertuples():
    laba_bersih = row.keuntungan - row.kerugian
    print(f"{row.tanggal} | {row.namaBisnis} | {format_rupiah(row.modal)} | {format_rupiah(row.pemasukan)} | "
          f"{format_rupiah(row.pengeluaran)} | {row.jamOperasional} | {row.status} | "
          f"{format_rupiah(row.keuntungan)} | {format_rupiah(row.kerugian)} | {format_rupiah(laba_bersih)}")

# === Rekomendasi Bisnis
print("\nRekomendasi Performa Bisnis:")

rekom_path = "randomForest/dataset/keputusanKata.csv"
if os.path.exists(rekom_path):
    df_rekom = pd.read_csv(rekom_path)
else:
    df_rekom = pd.DataFrame(columns=["tanggal", "keputusan", "kata"])

for row in df_new.itertuples():
    print(f"\nTanggal: {row.tanggal} | Bisnis: {row.namaBisnis}")

    hasil_rf = df_rekom[df_rekom['tanggal'] == row.tanggal] if 'tanggal' in df_rekom.columns else pd.DataFrame()

    if not hasil_rf.empty:
        for kalimat in hasil_rf['kata'].tolist():
            print(" -", kalimat)
    else:
        rekomendasi = []

        if row.status == 0 and row.keuntungan > 5000000:
            rekomendasi.append(random.choice(rekomendasi_variatif["status_untung_tapi_tak_sehat"]))
        elif row.status == 0:
            rekomendasi.append(random.choice(rekomendasi_variatif["status_tak_sehat"]))

        if row.jamOperasional > 10:
            rekomendasi.append(random.choice(rekomendasi_variatif["jam_berlebihan"]))

        if row.kerugian > 100000:
            rekomendasi.append(random.choice(rekomendasi_variatif["kerugian_tinggi"]))

        try:
            ratio = row.pengeluaran / (row.pemasukan if row.pemasukan != 0 else 1)
            if ratio > 0.8:
                rekomendasi.append(random.choice(rekomendasi_variatif["rasio_pengeluaran_tinggi"]))
        except:
            pass

        try:
            margin = (row.keuntungan - row.kerugian) / (row.pemasukan if row.pemasukan != 0 else 1)
            if margin < 0.05:
                rekomendasi.append(random.choice(rekomendasi_variatif["margin_tipis"]))
        except:
            pass

        if not rekomendasi:
            rekomendasi.append(random.choice(rekomendasi_variatif["kinerja_baik"]))

        for r in rekomendasi:
            print(" -", r)


