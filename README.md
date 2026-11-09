# Student Performance Prediction System

## Included files
- `student_data.csv` — ready-to-use practice CSV (395 records, Mathematics-style fields)
- `train_model.py` — preprocessing, Random Forest training, evaluation, and model saving
- `app.py` — Streamlit dashboard with CSV upload, column validation, predictions, and download
- `requirements.txt` — Python dependencies
- `SAD_REPORT.md` — SAD mini-analysis and reflection answers
- `Student_Performance_Colab.ipynb` — Google Colab notebook version

## Run locally (VS Code)
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open the local URL shown in the terminal (usually http://localhost:8501).

## Run in Google Colab
Open `Student_Performance_Colab.ipynb` in Colab. Upload `student_data.csv` when prompted, then run the cells in order. The notebook trains and evaluates the model.

## Target
`G3 < 10` = AT RISK; `G3 >= 10` = NOT AT RISK. `G3` is excluded from input features to prevent target leakage.

## Dataset note
The included CSV is a generated practice dataset with the same general column structure as the UCI Student Performance Mathematics data. It is included so the project can run immediately without downloading a dataset. It is **not** the original UCI records. If your instructor requires the authentic dataset, download the Mathematics file (`student-mat.csv`) from https://archive.ics.uci.edu/dataset/320/student+performance and replace the included CSV (save it as `student_data.csv`).

## GitHub
Create a repository named `student-performance-prediction`, then run:
```bash
git init
git add .
git commit -m "Complete student performance prediction system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. Never put passwords or access tokens in code.

## Responsible use
This is a classroom prototype. It is not a diagnosis, may not generalize to other institutions, and must not be the sole basis for decisions about students.
