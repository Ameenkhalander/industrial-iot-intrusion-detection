"""Ensemble Learning-Based Intrusion Detection for Industrial IoT - Streamlit app."""
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data import ATTACKS, LABEL_COL, PROTOCOLS, generate_synthetic_iiot, prepare_new, preprocess
from src.model import (all_models, feature_importance, results_table, split,
                       train_and_evaluate)

st.set_page_config(page_title="IIoT Intrusion Detection", page_icon="🛡️", layout="wide")
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ IIoT IDS")
st.sidebar.caption("Ensemble learning-based intrusion detection for Industrial IoT networks")

source = st.sidebar.radio("Data source", ["Synthetic IIoT traffic", "Upload CSV"])
mode = st.sidebar.radio("Task", ["Binary (Normal vs Attack)", "Multi-class (attack family)"])
multiclass = mode.startswith("Multi")
n_est = st.sidebar.slider("Trees / estimators per model", 50, 300, 100, 50)
test_size = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
selected = st.sidebar.multiselect("Models to train", list(all_models().keys()),
                                  default=list(all_models().keys()))

# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_synth(n, seed):
    return generate_synthetic_iiot(n, seed=seed)

df, target = None, LABEL_COL
if source == "Synthetic IIoT traffic":
    n_rows = st.sidebar.slider("Samples", 3000, 30000, 12000, 1000)
    df = load_synth(n_rows, int(seed))
else:
    up = st.sidebar.file_uploader("Labeled CSV (e.g. Edge-IIoTset, ToN_IoT, WUSTL-IIoT)", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        target = st.sidebar.selectbox("Label column", df.columns[::-1])
        drop_cols = st.sidebar.multiselect("Columns to drop (IDs, IPs, timestamps)", [c for c in df.columns if c != target])
        df = df.drop(columns=drop_cols)
        if len(df) > 100_000:
            df = df.sample(100_000, random_state=int(seed))
            st.sidebar.info("Sampled 100,000 rows to keep the app responsive.")

st.title("Ensemble Learning-Based Intrusion Detection for Industrial IoT")
if df is None:
    st.info("Upload a labeled CSV in the sidebar, or switch to the synthetic dataset.")
    st.stop()

tab_data, tab_train, tab_eval, tab_live = st.tabs(
    ["1 · Data & preprocessing", "2 · Train models", "3 · Evaluation", "4 · Live detection"])

# ------------------------------------------------------------------ tab 1
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))
    st.dataframe(df.head(50))

    left, right = st.columns(2)
    with left:
        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        df[target].value_counts().plot.barh(ax=ax, color="#0F6CBD")
        ax.set_xlabel("count")
        st.pyplot(fig)
    with right:
        st.subheader("Correlation (numeric features)")
        num = df.select_dtypes("number")
        if num.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            sns.heatmap(num.corr(), cmap="coolwarm", center=0, ax=ax, cbar=False)
            st.pyplot(fig)

    with st.expander("Preprocessing & feature engineering steps"):
        st.markdown(
            "1. Remove duplicates, replace ±inf, impute missing values (median / mode)\n"
            "2. Engineer features: packet ratio, byte ratio, total packets/bytes, "
            "average packet size, scan score (SYN ratio × failed-connection ratio)\n"
            "3. One-hot encode categorical columns (e.g. protocol)\n"
            "4. Standardize numeric features\n"
            "5. Stratified train/test split")

# ------------------------------------------------------------------ tab 2
with tab_train:
    st.write("Trains the baseline decision tree, four base learners and two ensembles "
             "(soft-voting and stacking), then compares them.")
    if st.button("🚀 Train & evaluate", type="primary", disabled=not selected):
        try:
            with st.spinner("Preprocessing..."):
                X, y, art = preprocess(df, target, multiclass)
                names = list(art["label"].classes_) if multiclass else ["Normal", "Attack"]
                Xtr, Xte, ytr, yte = split(X, y, test_size, int(seed))
            bar = st.progress(0.0, text="Training...")
            fitted, res = train_and_evaluate(
                Xtr, Xte, ytr, yte, multiclass, names, selected, int(seed), n_est,
                progress=lambda f, n: bar.progress(f, text=f"Finished {n}"))
            bar.empty()
            st.session_state.update(fitted=fitted, res=res, art=art, names=names,
                                    X_cols=list(X.columns), multiclass=multiclass)
            st.success("Training complete - see the Evaluation tab.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Training failed: {e}")
    if "res" in st.session_state:
        st.dataframe(results_table(st.session_state["res"]).style.format("{:.4f}").highlight_max(
            axis=0, color="#cfe8cf", subset=["Accuracy", "Precision", "Recall", "F1"]))

# ------------------------------------------------------------------ tab 3
with tab_eval:
    if "res" not in st.session_state:
        st.info("Train the models first.")
    else:
        res, fitted = st.session_state["res"], st.session_state["fitted"]
        names, mc = st.session_state["names"], st.session_state["multiclass"]
        table = results_table(res)
        st.subheader("Model comparison")
        metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1"] if m in table]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        table[metrics].plot.bar(ax=ax, rot=30)
        ax.set_ylim(max(0, table[metrics].min().min() - 0.03), 1.0)
        ax.legend(loc="lower right", ncol=4)
        st.pyplot(fig)

        pick = st.selectbox("Inspect model", list(res.keys()),
                            index=list(res.keys()).index(table.index[0]))
        r = res[pick]
        a, b = st.columns(2)
        with a:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(r["_cm"], annot=True, fmt="d", cmap="Blues", xticklabels=names,
                        yticklabels=names, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
        with b:
            if "_roc" in r:
                st.subheader("ROC curves")
                fig, ax = plt.subplots(figsize=(5, 4))
                for n, rr in res.items():
                    if "_roc" in rr:
                        ax.plot(*rr["_roc"], label=f"{n} ({rr['ROC-AUC']:.3f})")
                ax.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
                ax.legend(fontsize=7)
                st.pyplot(fig)
            else:
                st.subheader("Per-class metrics")
                rep = pd.DataFrame(r["_report"]).T.drop(columns="support", errors="ignore")
                st.dataframe(rep.style.format("{:.3f}"))

        imp = feature_importance(fitted[pick], st.session_state["X_cols"])
        if imp is not None:
            st.subheader("Feature importance")
            fig, ax = plt.subplots(figsize=(7, 4))
            imp.head(12)[::-1].plot.barh(ax=ax, color="#0F6CBD")
            st.pyplot(fig)

        best = st.selectbox("Model to export", list(fitted.keys()),
                            index=list(fitted.keys()).index(table.index[0]), key="export")
        buf = io.BytesIO()
        joblib.dump({"model": fitted[best], "art": st.session_state["art"], "names": names}, buf)
        st.download_button("⬇️ Download trained model (.joblib)", buf.getvalue(),
                           file_name="ids_model.joblib")

# ------------------------------------------------------------------ tab 4
with tab_live:
    if "fitted" not in st.session_state:
        st.info("Train the models first.")
    else:
        fitted, art = st.session_state["fitted"], st.session_state["art"]
        names, mc = st.session_state["names"], st.session_state["multiclass"]
        use = st.selectbox("Detector", list(fitted.keys()), key="live_model")
        model = fitted[use]

        st.subheader("Classify a batch of traffic (CSV)")
        batch = st.file_uploader("CSV with the same feature columns", type="csv", key="batch")
        if batch is not None:
            raw = pd.read_csv(batch)
            try:
                pred = model.predict(prepare_new(raw, art))
                out = raw.copy()
                out["prediction"] = [names[i] for i in pred] if mc else np.where(pred == 1, "Attack", "Normal")
                st.dataframe(out.head(200))
                st.bar_chart(out["prediction"].value_counts())
                st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not score this file: {e}")

        if source == "Synthetic IIoT traffic":
            st.subheader("Try a single flow")
            presets = {
                "Normal": dict(duration=2.0, src_pkts=18, dst_pkts=16, src_bytes=660, dst_bytes=780,
                               pkt_rate=12, syn_flag_ratio=0.08, unique_dst_ports=2,
                               payload_entropy=3.2, failed_conn_ratio=0.03, protocol="modbus"),
                "DDoS": dict(duration=0.3, src_pkts=220, dst_pkts=2, src_bytes=180, dst_bytes=40,
                             pkt_rate=810, syn_flag_ratio=0.75, unique_dst_ports=3,
                             payload_entropy=3.5, failed_conn_ratio=0.35, protocol="http"),
                "Port_Scan": dict(duration=0.12, src_pkts=3, dst_pkts=1, src_bytes=60, dst_bytes=30,
                                  pkt_rate=120, syn_flag_ratio=0.88, unique_dst_ports=85,
                                  payload_entropy=2.4, failed_conn_ratio=0.62, protocol="opcua"),
                "Ransomware": dict(duration=15, src_pkts=60, dst_pkts=50, src_bytes=3100,
                                   dst_bytes=1500, pkt_rate=24, syn_flag_ratio=0.07,
                                   unique_dst_ports=4, payload_entropy=7.3,
                                   failed_conn_ratio=0.08, protocol="http"),
            }
            p = presets[st.selectbox("Load preset", list(presets))]
            cols = st.columns(4)
            vals = {}
            for i, (k, v) in enumerate(p.items()):
                with cols[i % 4]:
                    if k == "protocol":
                        vals[k] = st.selectbox(k, PROTOCOLS, index=PROTOCOLS.index(v), key=f"{k}_{id(p)}")
                    else:
                        vals[k] = st.number_input(k, value=float(v), key=f"{k}_{id(p)}")
            if st.button("Detect"):
                row = pd.DataFrame([vals])
                pred = model.predict(prepare_new(row, art))[0]
                label = names[pred] if mc else ("Attack" if pred == 1 else "Normal")
                conf = ""
                if hasattr(model, "predict_proba"):
                    conf = f" (confidence {model.predict_proba(prepare_new(row, art)).max():.1%})"
                (st.success if label == "Normal" else st.error)(f"Prediction: **{label}**{conf}")
