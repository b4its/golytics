# =======================
# UKM Analytics
# =======================
program untuk meningkatkan bisnis di UMKM untuk bisa memprediksi keuntungan, dan kerugian mereka tiap minggu untuk kedepan, dan juga memberikan saran untuk meningkatkan performa bisnis.

## Requirements
-   Python 3.12.3
-   pip 25.1.1
-   setuptools 80.9.0
-   wheel 0.42.0
-   mlflow 2.9.0
-   pandas 2.2.0
-   numpy 1.26.3
-   torch 2.2.0
-   sentence-transformers 2.2.2

## Based Algorithm
-   Random Sampling / Randomization
-   Rule-Based / Conditional Logic
-   Date Arithmetic / Time Series Simulation

## Web Scrapping Technique
-   Regex Parsing
-   Rate Limiting
-   Data Cleaning / Heuristik
-   Logging & Error Handling


## Models Evaluation Algorithm
-   Mean Squared Error: MSE
-   Mean Absolute Error: MAE
-   R-squared metric R², or coefficient of determination: R²

## Classificiation Evaluation Algorithm
-   Confusion Matrix
-   Decision Tree

## Models
-   Multi Layer Perceptron
-   AutoRegressive Integrated Moving Average: Arima
-   Random Forest
-   Sentence Transformer


## how to run this models at you're environtment
- pip install packages
```bash
pip install -r requirements.txt
```

- generate csv data
```bash
cd generate
python generateJamOperasional.py
python generateDtPredict.py
```
- generate scrapping data for city and population density
```bash
python scrapCitizenCity.py
```
wait it until finished

- train the MLP models and Arima Forecast
```bash
├── dataset
│   ├── dtOperasional.csv
│   ├── dtPrediksi.csv
│   ├── hasilForecast_ARIMA.csv
│   └── hasilPrediksi.csv
├── generateDtPredict.py
└── generateJamOperasional.py
cd ../mlp
python trainProfit.py
cd ../randomForest
```

- generate recommendation words and train random forest models
```bash
.
├── recommendationPredict.py
└── trainOperasional.py

python recommendationPredict.py
python trainOperasional
cd ../
```

- run ml flow server:
```bash
    mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 \
    --port 5000
```

- Accuracy Test
```bash
python ujiAkurasiModel.py
```
- after run ujiAkurasiModel.py file, check the result in website:
```bash
localhost:5000
```
Accuracy Test for word variations
```bash
python ujiAkurasiKata.py
```
-   run the program
```bash
python main.py
```