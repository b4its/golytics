import csv
import random
import os
from datetime import datetime, timedelta

# Fungsi status operasional
def cek_status_operasional(jenisBisnis, jamOperasional, isWeekend):
    status = 1  # default sehat

    batas = {
        "F&B": (8, 12),
        "Digital & Teknologi": (1, 20),
        "Retail & Toko Serba Ada": (8, 10),
        "Fashion & Gaya Hidup": (8, 10),
        "Kantor Teknologi": (8, 10) if isWeekend else (1, 20),
        "Kesehatan & Kecantikan": (8, 10),
        "Pendidikan & Pelatihan": (1, 20),
        "Travel & Pariwisata": (8, 10),
        "Kerajinan & Produk Lokal": (8, 12),
        "Properti & Kontraktor": (8, 12),
    }

    minJam, maxJam = batas.get(jenisBisnis, (8, 10))

    if jamOperasional < minJam or jamOperasional > maxJam:
        status = 0  # tidak sehat

    return {
        "status": status
    }

# Konfigurasi
jenis_bisnis_list = [
    "F&B",
    "Digital & Teknologi",
    "Retail & Toko Serba Ada",
    "Fashion & Gaya Hidup",
    "Kreatif & Desain",
    "Kesehatan & Kecantikan",
    "Pendidikan & Pelatihan",
    "Travel & Pariwisata",
    "Kerajinan & Produk Lokal",
    "Properti & Kontraktor",
]
jumlah_data = 5000
entri_per_tanggal = 100  # 100 data per tanggal
interval_hari = 14  # tiap 2 minggu
start_date = datetime.today()

# Siapkan folder dan file output
os.makedirs("dataset", exist_ok=True)
output_file = "dataset/dtOperasional.csv"

# Simpan dataset
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["tanggal", "jenisBisnis", "jamOperasional", "isWeekend", "status"])

    for i in range(jumlah_data):
        jenis = random.choice(jenis_bisnis_list)
        jam = random.randint(1, 24)
        is_weekend = random.choice([True, False])

        tanggal = (start_date - timedelta(days=(i // entri_per_tanggal) * interval_hari)).strftime("%Y-%m-%d")
        status = cek_status_operasional(jenis, jam, is_weekend)["status"]

        writer.writerow([tanggal, jenis, jam, is_weekend, status])

print(f"✅ Dataset berhasil digenerate ke: {output_file}")




"""
note:
status:
0 | tidak sehat
1 | sehat
"""