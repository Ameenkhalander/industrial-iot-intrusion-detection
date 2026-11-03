"""Command-line training: python train.py [--csv path --target col] [--multiclass]"""
import argparse
import joblib
import pandas as pd

from src.data import LABEL_COL, generate_synthetic_iiot, preprocess
from src.model import results_table, split, train_and_evaluate

p = argparse.ArgumentParser()
p.add_argument("--csv", help="Path to a labeled IIoT CSV (default: synthetic data)")
p.add_argument("--target", default=LABEL_COL)
p.add_argument("--multiclass", action="store_true")
p.add_argument("--best", default="Voting (soft)", help="Model to save for the app")
a = p.parse_args()

df = pd.read_csv(a.csv) if a.csv else generate_synthetic_iiot()
X, y, art = preprocess(df, a.target, a.multiclass)
names = list(art["label"].classes_) if a.multiclass else ["Normal", "Attack"]
Xtr, Xte, ytr, yte = split(X, y)
fitted, res = train_and_evaluate(Xtr, Xte, ytr, yte, a.multiclass, names,
                                 progress=lambda f, n: print(f"{f:.0%} done: {n}"))
print(results_table(res).round(4).to_string())
joblib.dump({"model": fitted[a.best], "art": art, "names": names}, "models/ids_model.joblib")
print(f"Saved models/ids_model.joblib ({a.best})")
