import csv, random, os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Fungsi mapping jenis bisnis
def get_business_type(business_name):
    mapping = {
        "F&B": ["Rasa Pagi", "Sambal Senja", "Dapur Loka", "Teh & Cerita", "Jajan Jadoel"],
        "Digital & Teknologi": ["Jejak Digital", "Koding Karya", "Data Cerah", "BitNusa", "NusaCloud"],
        "Retail & Toko Serba Ada": ["Toko PastiAda", "Langgananku", "RumaRaya", "Harga Loka", "Belanja Pintar"],
        "Fashion & Gaya Hidup": ["Bumi Perca", "Hijab Pelita", "Langit Busana", "Rancak Raya", "Lini Rupa"],
        "Kreatif & Desain": ["Studio Karsa", "Citra Akar", "IdeLokal", "Bentuk Cerita", "Warna Rasa"],
        "Kesehatan & Kecantikan": ["Sehat Sentosa", "Rupa Sejiwa", "Laras Care", "Klinik Asa", "CantikNusa"],
        "Pendidikan & Pelatihan": ["Langkah Cerah", "Pelita Cita", "Kelas Rakyat", "Asa Cendekia", "Bimbingan Cerita"],
        "Travel & Pariwisata": ["Mitra Langit", "Jalan Jelajah", "NusaTrip", "Langkah Loka", "Tamasya Cerah"],
        "Kerajinan & Produk Lokal": ["Kriya Kita", "Rupa Nusantara", "Sentra Loka", "Karya Warna", "Loka Bambu"],
        "Properti & Kontraktor": ["Rancang Sejahtera", "Lahan Asa", "Rumah Kita", "Bangun Cerita", "Tata Griya"],
    }
    for jenis, names in mapping.items():
        if business_name in names:
            return jenis
    return "Lainnya"

# Batas jam operasional
batas = {
    "F&B": (8, 12),
    "Digital & Teknologi": (1, 20),
    "Retail & Toko Serba Ada": (8, 10),
    "Fashion & Gaya Hidup": (8, 10),
    "Kreatif & Desain": (1, 20),
    "Kesehatan & Kecantikan": (8, 10),
    "Pendidikan & Pelatihan": (1, 20),
    "Travel & Pariwisata": (8, 10),
    "Kerajinan & Produk Lokal": (8, 12),
    "Properti & Kontraktor": (8, 12),
    "Lainnya": (6, 12),
}

# Daftar nama bisnis
business_names = [
    "Rasa Pagi", "Sambal Senja", "Dapur Loka", "Teh & Cerita", "Jajan Jadoel",
    "Jejak Digital", "Koding Karya", "Data Cerah", "BitNusa", "NusaCloud",
    "Toko PastiAda", "Langgananku", "RumaRaya", "Harga Loka", "Belanja Pintar",
    "Bumi Perca", "Hijab Pelita", "Langit Busana", "Rancak Raya", "Lini Rupa",
    "Studio Karsa", "Citra Akar", "IdeLokal", "Bentuk Cerita", "Warna Rasa",
    "Sehat Sentosa", "Rupa Sejiwa", "Laras Care", "Klinik Asa", "CantikNusa",
    "Langkah Cerah", "Pelita Cita", "Kelas Rakyat", "Asa Cendekia", "Bimbingan Cerita",
    "Mitra Langit", "Jalan Jelajah", "NusaTrip", "Langkah Loka", "Tamasya Cerah",
    "Kriya Kita", "Rupa Nusantara", "Sentra Loka", "Karya Warna", "Loka Bambu",
    "Rancang Sejahtera", "Lahan Asa", "Rumah Kita", "Bangun Cerita", "Tata Griya"
]

# Generate alamat unik
def generate_unique_addresses(n):
    kota_list = [
        'Ambon', 'Balikpapan', 'Banda Aceh', 'Bandar Lampung', 'Bandung', 'Banjar', 'Banjarbaru',
        'Banjarmasin', 'Batam', 'Batu', 'Bekasi', 'Bengkulu', 'Binjai', 'Bitung', 'Bogor', 'Cilegon',
        'Cimahi', 'Cirebon', 'Denpasar', 'Depok', 'Dumai', 'Gorontalo', 'Jakarta', 'Jakarta Barat',
        'Jakarta Pusat', 'Jakarta Selatan', 'Jakarta Timur', 'Jakarta Utara', 'Jambi', 'Jayapura',
        'Kediri', 'Kendari', 'Kupang', 'Lhokseumawe', 'Lubuk Linggau', 'Madiun', 'Makassar', 'Malang',
        'Manado', 'Mataram', 'Medan', 'Padang', 'Padang Sidempuan', 'Palangka Raya', 'Palembang',
        'Palu', 'Pangkalpinang', 'Pasuruan', 'Pekalongan', 'Pekanbaru', 'Pematangsiantar',
        'Pontianak', 'Prabumulih', 'Probolinggo', 'Salatiga', 'Samarinda', 'Semarang', 'Serang',
        'Singkawang', 'Sorong', 'Sukabumi', 'Surabaya', 'Surakarta', 'Tangerang', 'Tangerang Selatan',
        'Tanjungpinang', 'Tarakan', 'Tasikmalaya', 'Tegal', 'Ternate', 'Yogyakarta'
    ]
    return [f"Jl. Pelaku Usaha No. {i}, {random.choice(kota_list)}" for i in range(1, n + 1)]

# Konfigurasi dasar
folder = "dataset"
os.makedirs(folder, exist_ok=True)
filename = os.path.join(folder, "dtPrediksi.csv")
jumlah_data = 11000
alamat_unik = generate_unique_addresses(jumlah_data)

# Periode tanggal ke depan
start_date = datetime.today()
max_date = start_date + relativedelta(months=11)
interval_hari = 14
entri_per_tanggal = 100

fieldnames = [
    "tanggal", "namaBisnis", "modal", "pemasukan", "pengeluaran", "jamOperasional",
    "jenisBisnis", "status", "keuntungan", "kerugian", "bisnisAddress", "isDigital", "hasWebsite"
]

# Tulis data ke CSV
with open(filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(jumlah_data):
        current_date = start_date + timedelta(days=(i // entri_per_tanggal) * interval_hari)
        if current_date > max_date:
            break  # Stop jika melewati 10 bulan dari hari ini

        tanggal = current_date.strftime("%Y-%m-%d")
        business_name = random.choice(business_names)
        jenis = get_business_type(business_name)
        jam_min, jam_max = batas.get(jenis, (6, 12))
        jamOperasional = random.randint(jam_min, jam_max)

        modal = round(random.uniform(1_000_000, 20_000_000), 2)
        pengeluaran = round(random.uniform(500_000, 5_000_000), 2)
        delta = round(random.uniform(-5_000_000, 10_000_000), 2)
        pemasukan = round(modal + delta, 2)
        hasil = round(pemasukan - modal, 2)

        status = 1 if hasil > 0 else 0
        keuntungan = round(hasil, 2) if hasil > 0 else 0.0
        kerugian = round(abs(hasil), 2) if hasil < 0 else 0.0

        row = {
            "tanggal": tanggal,
            "namaBisnis": business_name,
            "modal": modal,
            "pemasukan": pemasukan,
            "pengeluaran": pengeluaran,
            "jamOperasional": jamOperasional,
            "jenisBisnis": jenis,
            "status": status,
            "keuntungan": keuntungan,
            "kerugian": kerugian,
            "bisnisAddress": alamat_unik[i],
            "isDigital": jenis in ["Digital & Teknologi", "Kreatif & Desain"],
            "hasWebsite": random.choice([True, False])
        }

        writer.writerow(row)

print(f"✅ File '{filename}' berhasil dibuat dengan data dari hari ini sampai {max_date.strftime('%Y-%m-%d')}.")
