import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("GITHUB_ACTIONS") == "true":
    print("🌐 Mendeteksi GitHub Actions, mengalihkan tracking ke DagsHub Cloud...")
    
    token = os.environ.get("DAGSHUB_TOKEN", "")
    print(f"Token ada: {bool(token)}, panjang: {len(token)}")
    
    mlflow.set_tracking_uri(f"https://dagshub.com/satriaego/Kriteria-3-Membuat-Workflow-CI.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

mlflow.sklearn.autolog()

mlflow.set_experiment("Eksperimen_SML_Satria_Ego_Vania")

print(f"🚀 Memulai latihan model dengan n_estimators={args.n_estimators}, max_depth={args.max_depth}...")



data_path = os.path.join(current_dir, 'preprocessing', 'Titanic_cleaned_latest.csv')
    
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Waduh, file data tidak ditemukan di: {data_path}")
        
df = pd.read_csv(data_path)
df_numeric = df.select_dtypes(include=['int64', 'float64'])

X = df_numeric.drop(columns=['Survived'])
y = df_numeric['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=args.n_estimators, 
    max_depth=args.max_depth, 
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
mlflow.log_metric("testing_accuracy_score", akurasi)

print(f"✅ Model sukses dilatih! Akurasi Uji: {akurasi:.2%}")