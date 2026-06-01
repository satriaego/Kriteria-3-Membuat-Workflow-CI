import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
import mlflow
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("GITHUB_ACTIONS") == "true":
    import dagshub
    print("🌐 Mendeteksi GitHub Actions, mengalihkan tracking ke DagsHub Cloud...")
    dagshub.init(
        repo_owner='satriaego',
        repo_name='Kriteria-3-Membuat-Workflow-CI',
        mlflow=True,
    )
else:
    print("💻 Menjalankan secara lokal di komputer...")
    mlflow.set_tracking_uri(f"file:///{os.path.join(current_dir, 'mlruns')}")

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

model_output_path = os.path.join(current_dir, 'model.pkl')
joblib.dump(model, model_output_path)
print(f"📦 Artefak compiled model sukses dibuat di: {model_output_path}")

y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)[:, 1]

y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='binary')
train_loss = log_loss(y_train, y_train_proba)
train_prec = precision_score(y_train, y_train_pred, average='binary')
train_rec = recall_score(y_train, y_train_pred, average='binary')
train_roc = roc_auc_score(y_train, y_train_proba)

test_acc = accuracy_score(y_test, y_test_pred)

if "MLFLOW_RUN_ID" in os.environ:
    del os.environ["MLFLOW_RUN_ID"]

with mlflow.start_run():
    mlflow.log_metric("training_accuracy_score", train_acc)
    mlflow.log_metric("training_f1_score", train_f1)
    mlflow.log_metric("training_log_loss", train_loss)
    mlflow.log_metric("training_precision_score", train_prec)
    mlflow.log_metric("training_recall_score", train_rec)
    mlflow.log_metric("training_roc_auc", train_roc)
    mlflow.log_metric("training_score", train_acc)
    mlflow.log_metric("testing_accuracy_score", test_acc)

print(f"selesai")