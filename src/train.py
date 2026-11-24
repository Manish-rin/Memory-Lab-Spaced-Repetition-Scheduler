import time
import joblib
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEAT_PARQUET = PROJECT_ROOT / "data/processed/features_1m.parquet"
MODEL_OUT = PROJECT_ROOT / "models/syllabus_gb_model.joblib"

FEATURE_COLS = [
    "i",
    "d",
    "delta_t",
    "hist_len",
    "recall_count",
    "lapse_count",
    "lapse_ratio",
    "last_outcome",
    "recent_3_recall_rate",
    "last_interval",
    "mean_interval",
    "max_interval",
    "log_delta_t",
    "log_last_interval",
    "elapsed_to_last_ratio",
    "elapsed_to_max_ratio",
    "word_len",
]

def evaluate_predictions(name: str, y_true: np.ndarray, y_prob: np.ndarray, elapsed_sec: float):
    y_pred = (y_prob >= 0.5).astype(int)
    print(
        f"{name:34s} | "
        f"Log-Loss: {log_loss(y_true, y_prob):.4f} | "
        f"Brier: {brier_score_loss(y_true, y_prob):.4f} | "
        f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f} | "
        f"Acc: {accuracy_score(y_true, y_pred):.4f} | "
        f"Time: {elapsed_sec:5.1f}s"
    )

def train_and_evaluate():
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(FEAT_PARQUET)

    X = df.select(FEATURE_COLS).to_numpy()
    y = df["r"].to_numpy()
    groups = df["u"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train rows: {len(X_train):,} | Unseen-user Test rows: {len(X_test):,}\n")


    monotonic_constraints = [
        -1 if col in {"delta_t", "log_delta_t", "elapsed_to_last_ratio", "elapsed_to_max_ratio", "d"} else 0
        for col in FEATURE_COLS
    ]

    t0 = time.time()
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    lr.fit(X_train, y_train)
    evaluate_predictions("1. Logistic Regression", y_test, lr.predict_proba(X_test)[:, 1], time.time() - t0)


    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=50, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_predictions("2. Random Forest (100 trees)", y_test, rf.predict_proba(X_test)[:, 1], time.time() - t0)


    t0 = time.time()
    gb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=63,
        monotonic_cst=monotonic_constraints,
        random_state=42,
    )
    calibrated_gb = CalibratedClassifierCV(gb, method="isotonic", cv=3)
    calibrated_gb.fit(X_train, y_train)
    evaluate_predictions("3. Gradient Boosting (Calibrated)", y_test, calibrated_gb.predict_proba(X_test)[:, 1], time.time() - t0)

    joblib.dump({"model": calibrated_gb, "features": FEATURE_COLS}, MODEL_OUT)
    print(f"\nSaved final 3-model winner -> {MODEL_OUT}")

if __name__ == "__main__":
    train_and_evaluate()