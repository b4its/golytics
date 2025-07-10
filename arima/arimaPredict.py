import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import os

def forecast_arima(input_csv, output_csv):
    try:
        # Baca data
        df = pd.read_csv(input_csv)

        if 'keuntungan' not in df.columns or 'kerugian' not in df.columns:
            raise ValueError("Kolom keuntungan dan kerugian tidak ditemukan.")

        df['tanggal'] = pd.to_datetime(df['tanggal'])
        df['laba_bersih'] = df['keuntungan'] - df['kerugian']
        df = df.sort_values("tanggal")

        ts = df.groupby("tanggal")['laba_bersih'].mean()

        model_arima = ARIMA(ts, order=(2, 1, 2))
        arima_fit = model_arima.fit()
        forecast = arima_fit.forecast(steps=14)

        future_dates = [ts.index[-1] + timedelta(days=i + 1) for i in range(14)]
        df_forecast = pd.DataFrame({
            'tanggal': future_dates,
            'prediksi_laba_bersih': forecast
        })

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_forecast.to_csv(output_csv, index=False)
        print(f"✅ Forecast ARIMA disimpan di: {output_csv}")

    except Exception as e:
        print(f"❌ Gagal melakukan forecast ARIMA: {e}")
