from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction System")
st.caption("Data Mining Laboratory — Random Forest classification prototype")
st.warning("Classroom prototype only. Predictions are not diagnoses and must not be the sole basis for real student decisions.")

@st.cache_resource
def load_model():
    return joblib.load(ROOT/"student_model.pkl")

data_path = ROOT/"student_data.csv"
model_path = ROOT/"student_model.pkl"
if not data_path.exists() or not model_path.exists():
    st.error("Required files are missing. Keep student_data.csv in this folder and run: python train_model.py")
    st.stop()

data = pd.read_csv(data_path)
model = load_model()
a,b,c = st.columns(3)
a.metric("Dataset records", len(data))
b.metric("Average final grade", f"{data['G3'].mean():.2f}/20")
c.metric("Historical at-risk records", int((data["G3"]<10).sum()))

metrics_path = ROOT/"model_metrics.json"
if metrics_path.exists():
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Model evaluation")
    x1,x2,x3,x4 = st.columns(4)
    x1.metric("Accuracy", f"{m['accuracy']:.3f}")
    x2.metric("At-risk precision", f"{m['precision_at_risk']:.3f}")
    x3.metric("At-risk recall", f"{m['recall_at_risk']:.3f}")
    x4.metric("At-risk F1", f"{m['f1_at_risk']:.3f}")
    st.write("Confusion matrix (actual rows, predicted columns; order: Not at risk, At risk)")
    st.dataframe(pd.DataFrame(m["confusion_matrix"], index=["Actual not at risk","Actual at risk"], columns=["Predicted not at risk","Predicted at risk"]))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)
st.subheader("Predict uploaded student records")
st.info("Download the sample CSV, or upload a CSV containing all predictor columns. Do not include G3.")
st.download_button("Download sample input CSV", data=data.drop(columns=["G3"]).head(5).to_csv(index=False).encode(), file_name="sample_prediction_input.csv", mime="text/csv")
uploaded = st.file_uploader("Upload student CSV", type=["csv"])
if uploaded:
    try:
        incoming = pd.read_csv(uploaded)
        required = list(model.feature_names_in_)
        missing = [col for col in required if col not in incoming.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif incoming.empty:
            st.error("The uploaded CSV is empty.")
        else:
            pred = model.predict(incoming[required])
            results = incoming.copy()
            results["Prediction"] = ["AT RISK" if int(v)==1 else "NOT AT RISK" for v in pred]
            st.success(f"Predictions completed for {len(results)} records.")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted at-risk records", int((pred==1).sum()))
            st.download_button("Download prediction results", data=results.to_csv(index=False).encode(), file_name="student_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Unable to read or predict this CSV: {e}")
st.subheader("System process")
st.write("Upload CSV → Validate columns → Preprocess → Predict risk → Display and export results")
