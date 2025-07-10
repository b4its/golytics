import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import mlflow
import mlflow.pytorch
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

# === MLP Model ===
class ProfitMLP(nn.Module):
    def __init__(self, input_dim):
        super(ProfitMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# === Load Dataset ===
data_path = "generate/dataset/dtPrediksi.csv"
df = pd.read_csv(data_path)

required_cols = ['modal', 'pemasukan', 'pengeluaran', 'jamOperasional',
                 'jenisBisnis', 'keuntungan', 'kerugian', 'tanggal']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di {data_path}")

# Preprocessing
float_cols = ['modal', 'pemasukan', 'pengeluaran', 'keuntungan', 'kerugian']
df[float_cols] = df[float_cols].astype(np.float64)
df[['modal', 'pemasukan', 'pengeluaran', 'jamOperasional']] = df[['modal', 'pemasukan', 'pengeluaran', 'jamOperasional']].clip(lower=0)

# Encode
label_encoder = LabelEncoder()
df['jenis_encoded'] = label_encoder.fit_transform(df['jenisBisnis'])
df['laba_bersih'] = df['keuntungan'] - df['kerugian']

# Features & Target
X = df[['modal', 'pemasukan', 'pengeluaran', 'jamOperasional', 'jenis_encoded']]
y = df[['keuntungan', 'kerugian']]
y_log = np.log1p(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("mlp/scalers", exist_ok=True)
joblib.dump(scaler, "mlp/scalers/profit_scaler.pkl")

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProfitMLP(input_dim=5).to(device).double()

# Training Setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float64).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float64).to(device)

# === MLflow Config ===
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Profit Forecasting")

with mlflow.start_run(run_name="MLP-RF-ARIMA Evaluation"):

    # Log Params
    mlflow.log_param("mlp_epochs", 1000)
    mlflow.log_param("mlp_lr", 0.001)
    mlflow.log_param("rf_estimators", 100)

    print("Melatih model ProfitMLP...")
    for epoch in range(1000):
        model.train()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} - Loss: {loss.item():.6f}")

    # Simpan model
    os.makedirs("mlp/models", exist_ok=True)
    torch.save(model.state_dict(), "mlp/models/profit_predictor.pth")

    example_input = np.random.randn(1, 5).astype(np.float32)
    mlflow.pytorch.log_model(model, artifact_path="MLP_Model", input_example=example_input)
    print("Model ProfitMLP berhasil disimpan.")

    # === Evaluasi MLP ===
    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float64).to(device)
    with torch.no_grad():
        y_pred_log = model(X_test_tensor).cpu().numpy()

    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test.values)

    mse_mlp = mean_squared_error(y_true, y_pred)
    mae_mlp = mean_absolute_error(y_true, y_pred)
    r2_mlp = r2_score(y_true, y_pred)

    print("\nEvaluasi Model ProfitMLP:")
    print(f"MSE : {mse_mlp:.4f}")
    print(f"MAE : {mae_mlp:.4f}")
    print(f"R²  : {r2_mlp:.4f}")

    mlflow.log_metric("MLP_MSE", mse_mlp)
    mlflow.log_metric("MLP_MAE", mae_mlp)
    mlflow.log_metric("MLP_R2", r2_mlp)

    for i, target in enumerate(['Keuntungan', 'Kerugian']):
        mse_i = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2_i = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{target} - MSE: {mse_i:.4f}, MAE: {mae_i:.4f}, R²: {r2_i:.4f}")
        mlflow.log_metric(f"MLP_{target}_MSE", mse_i)
        mlflow.log_metric(f"MLP_{target}_MAE", mae_i)
        mlflow.log_metric(f"MLP_{target}_R2", r2_i)

    # === Random Forest ===
    print("\nEvaluasi Model Random Forest:")
    rf_keuntungan = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_kerugian = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_keuntungan.fit(X_train, np.expm1(y_train['keuntungan']))
    rf_kerugian.fit(X_train, np.expm1(y_train['kerugian']))

    mlflow.sklearn.log_model(rf_keuntungan, "RF_Keuntungan_Model")
    mlflow.sklearn.log_model(rf_kerugian, "RF_Kerugian_Model")

    y_pred_rf_keuntungan = rf_keuntungan.predict(X_test)
    y_pred_rf_kerugian = rf_kerugian.predict(X_test)

    y_true_rf_keuntungan = np.expm1(y_test['keuntungan'])
    y_true_rf_kerugian = np.expm1(y_test['kerugian'])

    for name, y_true_rf, y_pred_rf in [
        ("Keuntungan", y_true_rf_keuntungan, y_pred_rf_keuntungan),
        ("Kerugian", y_true_rf_kerugian, y_pred_rf_kerugian)
    ]:
        mse = mean_squared_error(y_true_rf, y_pred_rf)
        mae = mean_absolute_error(y_true_rf, y_pred_rf)
        r2 = r2_score(y_true_rf, y_pred_rf)
        print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        mlflow.log_metric(f"RF_{name}_MSE", mse)
        mlflow.log_metric(f"RF_{name}_MAE", mae)
        mlflow.log_metric(f"RF_{name}_R2", r2)

    # === ARIMA ===
    print("\nEvaluasi Model ARIMA:")

    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['laba_bersih'] = df['keuntungan'] - df['kerugian']
    df_ts = df.set_index('tanggal').resample('W').sum()[['laba_bersih']].dropna()

    print(f"Jumlah data setelah resample: {len(df_ts)}")

    if len(df_ts) < 6:
        print("Data terlalu sedikit untuk evaluasi ARIMA.")
    else:
        try:
            ts_data = df_ts['laba_bersih']
            train_size = int(len(ts_data) * 0.8)
            ts_train, ts_test = ts_data[:train_size], ts_data[train_size:]

            arima_model = ARIMA(ts_train, order=(1, 1, 1)).fit()
            arima_forecast = arima_model.forecast(steps=len(ts_test))

            mse_arima = mean_squared_error(ts_test, arima_forecast)
            mae_arima = mean_absolute_error(ts_test, arima_forecast)
            r2_arima = r2_score(ts_test, arima_forecast)

            print(f"ARIMA - MSE: {mse_arima:.4f}, MAE: {mae_arima:.4f}, R²: {r2_arima:.4f}")
            mlflow.log_metric("ARIMA_MSE", mse_arima)
            mlflow.log_metric("ARIMA_MAE", mae_arima)
            mlflow.log_metric("ARIMA_R2", r2_arima)
        except Exception as e:
            print(f"Gagal ARIMA: {e}")
            mlflow.log_param("ARIMA_Error", str(e))
