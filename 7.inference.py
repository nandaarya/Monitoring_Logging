import pandas as pd
import requests
import joblib

# Mapping status gizi dari preprocessing dataset
mapping_status_gizi = {
    'normal': 0,
    'stunted': 1,
    'severely stunted': 2,
    'tinggi': 3
}

def preprocess_input(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    # Hilangkan duplikat (jika ada)
    df = df.drop_duplicates()

    # Ubah semua string di 'Jenis Kelamin' menjadi lowercase untuk konsistensi
    df['Jenis Kelamin'] = df['Jenis Kelamin'].str.lower()

    # Binary encoding manual: 1 jika perempuan, 0 jika laki-laki
    df['Jenis Kelamin_perempuan'] = (df['Jenis Kelamin'] == 'perempuan').astype(int)

    # Drop kolom asli
    df = df.drop(columns=['Jenis Kelamin'])

    # Fitur numerik
    num_features = ['Umur (bulan)', 'Tinggi Badan (cm)']
    df[num_features] = df[num_features].astype(float)

    # Load scaler yang sudah di-fit saat training
    scaler = joblib.load(scaler_path)

    # Transformasi fitur numerik
    df[num_features] = scaler.transform(df[num_features])

    # Drop kolom target
    if 'Status Gizi' in df.columns:
        df = df.drop(columns=['Status Gizi'])

    return df

def predict(input_df: pd.DataFrame, mlflow_url: str):
    payload = {
        "dataframe_split": {
            "columns": input_df.columns.tolist(),
            "data": input_df.values.tolist()
        }
    }

    response = requests.post(
        mlflow_url + "/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

    preds_response = response.json()
    print("Raw prediksi dari MLflow:", preds_response)

    preds_numeric = preds_response.get('predictions', [])

    if not isinstance(preds_numeric, list):
        preds_numeric = [preds_numeric]

    # Mapping balik ke label string
    inverse_mapping = {v: k for k, v in mapping_status_gizi.items()}
    preds_label = [inverse_mapping.get(pred, "unknown") for pred in preds_numeric]

    return preds_label

if __name__ == "__main__":
    # Contoh input data
    data = {
        "Umur (bulan)": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Tinggi Badan (cm)": [45, 50, 55, 58, 60, 62, 65, 70, 75, 80, 85, 90, 55],
        "Jenis Kelamin": [
            "perempuan", "laki-laki", "perempuan", "laki-laki", "perempuan", "laki-laki",
            "perempuan", "laki-laki", "perempuan", "laki-laki", "perempuan", "laki-laki", "laki-laki"
        ]
    }
    input_df = pd.DataFrame(data)

    # Path ke scaler dari training
    scaler_path = "scaler.pkl"

    # Preprocessing input
    input_processed = preprocess_input(input_df, scaler_path)
    print("Data setelah preprocessing:")
    print(input_processed)

    # URL MLflow Model Serve
    mlflow_url = "http://127.0.0.1:8000"

    # Prediksi
    preds = predict(input_processed, mlflow_url)
    print("Prediksi status gizi:", preds)