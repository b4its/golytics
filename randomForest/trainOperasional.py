import os, joblib, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# === Model Status MLP ===
class StatusMLP(nn.Module):
    def __init__(self, input_dim):
        super(StatusMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === Load dataset
df = pd.read_csv("../generate/dataset/dtOperasional.csv")

# === Encode jenisBisnis
le = LabelEncoder()
df['jenis_encoded'] = le.fit_transform(df['jenisBisnis'])

# Konversi isWeekend ke integer
df['isWeekend'] = df['isWeekend'].astype(int)

# === Fitur dan Label
X = df[['jenis_encodejamOperasionald', '', 'isWeekend']]
y = df['status']

# === Scaling untuk MLP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan encoder dan scaler ke `mlp/scalers`
os.makedirs("scalers", exist_ok=True)
joblib.dump(le, "scalers/status_label_encoder.pkl")
joblib.dump(scaler, "scalers/status_scaler.pkl")

# === Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =============================
# === Train Random Forest ====
# =============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Akurasi Random Forest: {acc_rf*100:.2f}%")

# Simpan model RF ke `mlp/models`
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/status_rf_model.pkl")
print("Model Random Forest disimpan ke: models/status_rf_model.pkl")

# =============================
# === Train MLP (Optional) ====
# =============================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

model_mlp = StatusMLP(input_dim=3)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=0.001)

print("Melatih model StatusMLP...")
for epoch in range(500):
    model_mlp.train()
    output = model_mlp(X_train_tensor)
    loss = criterion(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# Simpan model MLP ke `mlp/models`
torch.save(model_mlp.state_dict(), "models/status_predictor.pth")
print("Model StatusMLP disimpan ke: models/status_predictor.pth")
