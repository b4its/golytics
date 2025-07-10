import os, joblib, pandas as pd, numpy as np, torch, torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# === MLP Model ===
class ProfitMLP(nn.Module):
    def __init__(self, input_dim):
        super(ProfitMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Output: [keuntungan, kerugian]
        )

    def forward(self, x):
        return self.net(x)

# === Load Dataset ===
data_path = "../generate/dataset/dtPrediksi.csv"
df = pd.read_csv(data_path)

required_cols = ['modal', 'pemasukan', 'pengeluaran', 'jamOperasional',
                 'jenisBisnis', 'keuntungan', 'kerugian', 'tanggal']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Kolom '{col}' tidak ditemukan di {data_path}")

# Konversi numerik dan penanganan nilai negatif
float_cols = ['modal', 'pemasukan', 'pengeluaran', 'keuntungan', 'kerugian']
df[float_cols] = df[float_cols].astype(np.float32)
df[float_cols] = df[float_cols].applymap(lambda x: max(x, 0))

# Encode jenisBisnis
label_encoder = LabelEncoder()
df['jenis_encoded'] = label_encoder.fit_transform(df['jenisBisnis'])

# Tambahkan laba_bersih untuk ARIMA
df['laba_bersih'] = df['keuntungan'] - df['kerugian']

# === Feature & Target
X = df[['modal', 'pemasukan', 'pengeluaran', 'jamOperasional', 'jenis_encoded']]
y = df[['keuntungan', 'kerugian']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scaling
input_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

# === Simpan Scaler ke mlp/scalers/
os.makedirs("scalers", exist_ok=True)
joblib.dump(input_scaler, "scalers/profit_scaler.pkl")
joblib.dump(target_scaler, "scalers/profit_target_scaler.pkl")

# === Training Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProfitMLP(input_dim=5).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

print("� Melatih model ProfitMLP...")
for epoch in range(500):
    model.train()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# === Simpan Model ke mlp/models/
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/profit_predictor.pth")
print("✅ Model ProfitMLP berhasil disimpan.")

# === Forecast ARIMA
print("\n� Membuat prediksi laba bersih (ARIMA)...")

df['tanggal'] = pd.to_datetime(df['tanggal'])
df = df.sort_values("tanggal")

# Gunakan resample agar freq terdefinisi
ts = df.set_index("tanggal")['laba_bersih'].resample('W').mean()
ts = ts.dropna()

try:
    # Gunakan order sederhana agar stabil untuk dataset kecil
    model_arima = ARIMA(ts, order=(1, 1, 1))
    arima_fit = model_arima.fit()
    forecast = arima_fit.forecast(steps=14)

    # Buat tanggal ke depan sesuai frekuensi mingguan
    future_dates = pd.date_range(start=ts.index[-1] + timedelta(days=7), periods=14, freq='W')
    df_forecast = pd.DataFrame({
        'tanggal': future_dates,
        'prediksi_laba_bersih': forecast
    })

    os.makedirs("../arima/forecast", exist_ok=True)
    df_forecast.to_csv("../arima/forecast/hasilForecast_ARIMA.csv", index=False)
    print("✅ Forecast ARIMA disimpan di: arima/forecast/hasilForecast_ARIMA.csv")

except Exception as e:
    print(f"❌ Gagal membuat forecast ARIMA: {e}")
