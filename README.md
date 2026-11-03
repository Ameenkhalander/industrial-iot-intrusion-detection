# Ensemble Learning-Based Intrusion Detection for Industrial IoT

Streamlit app that detects cyber attacks in Industrial IoT network traffic using ensemble learning.

## What it does
- **Preprocessing & feature engineering**: dedupe, impute, packet/byte ratios, scan score, one-hot encoding, scaling
- **Models**: Decision Tree (baseline), Random Forest, Extra Trees, Gradient Boosting, AdaBoost, **Soft Voting ensemble**, **Stacking ensemble**
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, False Alarm Rate, confusion matrix, ROC curves, feature importance, latency
- **Modes**: Binary (Normal vs Attack) or Multi-class (DDoS, Port Scan, MITM, Injection, Ransomware)
- **Live detection**: score a CSV batch or a single flow; export the trained model

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo
2. share.streamlit.io -> New app -> select repo, branch, main file `app.py`

## Use a real dataset
Choose **Upload CSV** in the sidebar, pick the label column and drop ID/IP/timestamp columns.
Suitable datasets: Edge-IIoTset, ToN_IoT, WUSTL-IIoT-2021, X-IIoTID.
(Engineered features only apply when columns named like the synthetic ones exist; other columns are used as-is.)

## CLI training
```bash
python train.py                         # synthetic data, binary
python train.py --multiclass
python train.py --csv data.csv --target label
```

## Structure
```
app.py            Streamlit UI
train.py          CLI training script
src/data.py       synthetic data, cleaning, feature engineering, preprocessing
src/model.py      models, ensembles, metrics
data/             sample CSV
```
