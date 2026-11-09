from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_data.csv"

def main():
    # Uses the included CSV so the project runs immediately.
    # To use the official UCI dataset instead, download student-mat.csv from the UCI repository
    # and replace student_data.csv with its contents.
    if not DATA_PATH.exists():
        raise FileNotFoundError("student_data.csv is missing. Keep it in the same folder as train_model.py.")
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    if "G3" not in df.columns:
        raise ValueError("CSV must contain G3 final Mathematics grade.")
    print("Rows and columns:", df.shape)
    print("\\nFirst five rows:\\n", df.head())
    print("\\nData types:\\n", df.dtypes)
    print("\\nMissing values:\\n", df.isnull().sum())

    y = (df["G3"] < 10).astype(int)
    X = df.drop(columns=["G3"])  # G3 is the target source, never a predictor.
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipe, numeric_cols), ("categorical", categorical_pipe, categorical_cols)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_at_risk": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
        "recall_at_risk": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
        "f1_at_risk": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0,1]).tolist(),
        "classification_report": classification_report(y_test, pred, labels=[0,1], target_names=["Not at risk","At risk"], output_dict=True, zero_division=0)
    }
    joblib.dump(model, ROOT/"student_model.pkl")
    (ROOT/"model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    X.head(5).to_csv(ROOT/"sample_prediction_input.csv", index=False)
    print("\\nAccuracy:", round(metrics["accuracy"],3))
    print(classification_report(y_test, pred, labels=[0,1], target_names=["Not at risk","At risk"], zero_division=0))
    print("Confusion matrix:", metrics["confusion_matrix"])
    print("\\nCreated student_model.pkl, model_metrics.json, sample_prediction_input.csv")

if __name__ == "__main__":
    main()
