"""Data loading, synthetic IIoT traffic generation and preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ATTACKS = ["DDoS", "Port_Scan", "MITM", "Injection", "Ransomware"]
PROTOCOLS = ["modbus", "mqtt", "opcua", "dnp3", "http"]
LABEL_COL = "attack_type"


def generate_synthetic_iiot(n_samples: int = 12000, attack_ratio: float = 0.35,
                            seed: int = 42) -> pd.DataFrame:
    """Create a realistic-looking Industrial IoT network-flow dataset.

    Each attack class has a distinct (but overlapping) statistical profile so
    the models have a genuine learning problem. Replace with a real dataset
    (Edge-IIoTset, ToN_IoT, WUSTL-IIoT, X-IIoTID) by uploading a CSV.
    """
    rng = np.random.default_rng(seed)
    n_att = int(n_samples * attack_ratio)
    n_norm = n_samples - n_att
    per_att = np.full(len(ATTACKS), n_att // len(ATTACKS))
    per_att[0] += n_att - per_att.sum()

    def block(n, label, dur, pkts, byt, rate, syn, ports, ent, fail, proto_p):
        return pd.DataFrame({
            "duration": rng.gamma(dur[0], dur[1], n),
            "src_pkts": rng.poisson(pkts[0], n) + 1,
            "dst_pkts": rng.poisson(pkts[1], n) + 1,
            "src_bytes": rng.gamma(byt[0], byt[1], n),
            "dst_bytes": rng.gamma(byt[2], byt[3], n),
            "pkt_rate": rng.gamma(rate[0], rate[1], n),
            "syn_flag_ratio": np.clip(rng.normal(syn[0], syn[1], n), 0, 1),
            "unique_dst_ports": rng.poisson(ports, n) + 1,
            "payload_entropy": np.clip(rng.normal(ent[0], ent[1], n), 0, 8),
            "failed_conn_ratio": np.clip(rng.normal(fail[0], fail[1], n), 0, 1),
            "protocol": rng.choice(PROTOCOLS, n, p=proto_p),
            LABEL_COL: label,
        })

    parts = [block(n_norm, "Normal", (2, 1.5), (18, 16), (3, 220, 3, 260),
                   (2, 6), (0.08, 0.05), 2, (3.2, 0.7), (0.03, 0.03),
                   [0.35, 0.3, 0.2, 0.1, 0.05])]
    profiles = {
        "DDoS": ((1, 0.3), (220, 2), (2, 90, 1, 40), (9, 90), (0.75, 0.15), 3,
                 (3.5, 0.9), (0.35, 0.15), [0.1, 0.1, 0.05, 0.05, 0.7]),
        "Port_Scan": ((0.6, 0.2), (3, 1), (1, 60, 1, 30), (3, 40), (0.88, 0.08),
                      85, (2.4, 0.7), (0.62, 0.15), [0.2, 0.2, 0.2, 0.2, 0.2]),
        "MITM": ((4, 2.2), (30, 28), (3, 300, 3, 280), (2, 7), (0.1, 0.06), 3,
                 (4.6, 0.9), (0.06, 0.04), [0.3, 0.3, 0.2, 0.15, 0.05]),
        "Injection": ((2, 1.6), (14, 10), (4, 350, 2, 150), (2, 6), (0.09, 0.05),
                      2, (5.8, 0.8), (0.1, 0.07), [0.45, 0.2, 0.15, 0.1, 0.1]),
        "Ransomware": ((6, 2.6), (60, 50), (6, 520, 4, 380), (2, 12), (0.07, 0.05),
                       4, (7.3, 0.5), (0.08, 0.06), [0.05, 0.05, 0.1, 0.1, 0.7]),
    }
    for (name, p), n in zip(profiles.items(), per_att):
        parts.append(block(n, name, *p))

    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    df["is_attack"] = (df[LABEL_COL] != "Normal").astype(int)
    # sprinkle missing values so preprocessing has real work to do
    for i, col in enumerate(["payload_entropy", "failed_conn_ratio", "duration"]):
        df.loc[df.sample(frac=0.01, random_state=seed + i).index, col] = np.nan
    return df.reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, replace inf, impute missing values (median / mode)."""
    df = df.drop_duplicates().replace([np.inf, -np.inf], np.nan).copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features when the standard columns exist."""
    df = df.copy()
    c = set(df.columns)
    if {"src_pkts", "dst_pkts"} <= c:
        df["pkt_ratio"] = df["src_pkts"] / (df["dst_pkts"] + 1)
        df["total_pkts"] = df["src_pkts"] + df["dst_pkts"]
    if {"src_bytes", "dst_bytes"} <= c:
        df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
        df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
    if {"total_bytes", "total_pkts"} <= set(df.columns):
        df["avg_pkt_size"] = df["total_bytes"] / (df["total_pkts"] + 1)
    if {"syn_flag_ratio", "failed_conn_ratio"} <= c:
        df["scan_score"] = df["syn_flag_ratio"] * df["failed_conn_ratio"]
    return df


def _encode_and_scale(X: pd.DataFrame, art: dict | None):
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, dtype=float)
    if art is None:  # fit
        feature_cols = X.columns.tolist()
        num_cols = [c for c in feature_cols if X[c].nunique() > 2]
        scaler = StandardScaler().fit(X[num_cols])
        art = {"feature_cols": feature_cols, "num_cols": num_cols, "scaler": scaler}
    else:
        X = X.reindex(columns=art["feature_cols"], fill_value=0.0)
    X[art["num_cols"]] = art["scaler"].transform(X[art["num_cols"]])
    return X, art


def preprocess(df: pd.DataFrame, target: str, multiclass: bool = False):
    """Clean, engineer features, encode and scale. Returns X, y, artefacts.

    target     -> column holding the label (string or 0/1)
    multiclass -> keep attack families; otherwise binary Normal vs Attack
    """
    df = engineer_features(clean(df))
    y_raw = df[target]
    if multiclass:
        le = LabelEncoder().fit(y_raw.astype(str))
        y = le.transform(y_raw.astype(str))
    else:
        le = None
        if pd.api.types.is_numeric_dtype(y_raw):
            y = (y_raw != 0).astype(int).to_numpy()
        else:
            benign = y_raw.astype(str).str.lower().isin(["normal", "benign", "0"])
            y = (~benign).astype(int).to_numpy()

    X = df.drop(columns=[c for c in (target, "is_attack", LABEL_COL) if c in df.columns])
    X, art = _encode_and_scale(X, None)
    art.update({"label": le, "multiclass": multiclass, "target": target})
    return X, y, art


def prepare_new(df_raw: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Apply a fitted preprocessing pipeline to new, unlabeled rows."""
    df = engineer_features(df_raw.replace([np.inf, -np.inf], np.nan))
    df = df.drop(columns=[c for c in (art["target"], "is_attack", LABEL_COL) if c in df.columns])
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode().iloc[0])
    X, _ = _encode_and_scale(df, art)
    return X
