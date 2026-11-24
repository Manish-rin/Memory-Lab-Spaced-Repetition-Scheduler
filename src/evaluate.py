import joblib
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEAT_PARQUET = PROJECT_ROOT / "data/processed/features_1m.parquet"
MODEL_PATH = PROJECT_ROOT / "models/syllabus_gb_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"


def extract_single_row(d: float, t_hist: str, r_hist: str, delta_t: int, word: str) -> dict:
    t_list = [float(x) for x in t_hist.split(",")]
    r_list = [int(x) for x in r_hist.split(",")]
    hist_len = len(r_list)
    recall_count = sum(r_list)
    lapse_count = hist_len - recall_count
    last_interval = t_list[-1]
    max_interval = max(t_list)

    return {
        "i": hist_len + 1,
        "d": d,
        "delta_t": delta_t,
        "hist_len": hist_len,
        "recall_count": recall_count,
        "lapse_count": lapse_count,
        "lapse_ratio": lapse_count / hist_len,
        "last_outcome": r_list[-1],
        "recent_3_recall_rate": float(np.mean(r_list[-3:])),
        "last_interval": last_interval,
        "mean_interval": float(np.mean(t_list)),
        "max_interval": max_interval,
        "log_delta_t": float(np.log1p(delta_t)),
        "log_last_interval": float(np.log1p(last_interval)),
        "elapsed_to_last_ratio": delta_t / (last_interval + 0.1),
        "elapsed_to_max_ratio": delta_t / (max_interval + 0.1),
        "word_len": len(word),
    }


def sweep_forgetting_curve(
    model,
    feature_cols: list,
    d: float,
    t_hist: str,
    r_hist: str,
    word: str,
    max_days: int = 60,
):
    days = np.arange(1, max_days + 1)
    df_sweep = pd.DataFrame(
        [extract_single_row(d, t_hist, r_hist, int(dt), word) for dt in days]
    )[feature_cols]
    probs = model.predict_proba(df_sweep.to_numpy())[:, 1]
    return days, probs


def generate_diagnostics():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    cal_model = bundle["model"]
    feature_cols = bundle["features"]


    df = pl.read_parquet(FEAT_PARQUET)
    X = df.select(feature_cols).to_numpy()
    y = df["r"].to_numpy()
    groups = df["u"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(splitter.split(X, y, groups))
    X_test, y_test = X[test_idx], y[test_idx]

    y_prob = cal_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print(
        f"Verified Holdout Test ({len(X_test):,} unseen-user rows) -> "
        f"Log-Loss: {log_loss(y_test, y_prob):.4f} | "
        f"Brier: {brier_score_loss(y_test, y_prob):.4f} | "
        f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f} | "
        f"Acc: {accuracy_score(y_test, y_pred):.4f}"
    )

    # 2. Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=12, strategy="uniform")
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=2,
        color="#0d9488",
        label="Calibrated Gradient Boosting",
    )
    plt.title("Probability Calibration Curve (Unseen-User Test Set)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Recall Probability")
    plt.ylabel("Observed Empirical Recall Fraction")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "calibration_curve.png", dpi=200)
    plt.close()

    perm = permutation_importance(
        cal_model,
        X_test[:15_000],
        y_test[:15_000],
        scoring="neg_log_loss",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    sorted_idx = np.argsort(perm.importances_mean)
    plt.figure(figsize=(8.5, 6))
    plt.barh(
        np.array(feature_cols)[sorted_idx],
        perm.importances_mean[sorted_idx],
        xerr=perm.importances_std[sorted_idx],
        color="#0284c7",
        alpha=0.9,
    )
    plt.title("Permutation Feature Importance (Impact on Log-Loss)", fontsize=12, fontweight="bold")
    plt.xlabel("Increase in Log-Loss When Feature Shuffled (Higher = More Important)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=200)
    plt.close()

    scenarios = [
        ("Easy Word + Clean Streak (d=1, '0,1,3,7' / '1,1,1,1')", "apple", 1.0, "0,1,3,7", "1,1,1,1", "#0284c7"),
        ("Hard Word + Clean Streak (d=9, '0,1,3,7' / '1,1,1,1')", "ephemeral", 9.0, "0,1,3,7", "1,1,1,1", "#0d9488"),
        ("Recent Lapse Penalty (d=5, '0,1,3,7' / '1,1,1,0')", "abandon", 5.0, "0,1,3,7", "1,1,1,0", "#e11d48"),
        ("Cold Start First Review (d=5, '0' / '1')", "civilian", 5.0, "0", "1", "#f59e0b"),
    ]

    plt.figure(figsize=(9.5, 5.5))
    target_retention = 0.85
    plt.axhline(
        target_retention,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Target Retention Threshold (85%)",
    )

    print("\n=== Live Counterfactual Stress Test Results (Target Retention = 85%) ===")
    for label, word, d, t_hist, r_hist, color in scenarios:
        days, probs = sweep_forgetting_curve(
            cal_model, feature_cols, d, t_hist, r_hist, word, max_days=60
        )
        safe_days = days[probs >= target_retention]
        opt_day = int(safe_days[-1]) if len(safe_days) > 0 else 1
        print(
            f"  {label:56s} -> Optimal Next Review: Day {opt_day:2d} | "
            f"P(1d)={probs[0]:.3f}, P(7d)={probs[6]:.3f}, P(30d)={probs[29]:.3f}"
        )
        plt.plot(
            days,
            probs,
            linewidth=2.2,
            color=color,
            label=f"{label} [Next: Day {opt_day}]",
        )

    plt.title("Model-Predicted Forgetting Curves Across Memory Histories", fontsize=12, fontweight="bold")
    plt.xlabel("Days Elapsed Since Last Review (delta_t)")
    plt.ylabel("Predicted Recall Probability P(recall)")
    plt.ylim(0.35, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "forgetting_curves.png", dpi=200)
    plt.close()
    print(f"\nSaved all 3 diagnostic plots -> {REPORTS_DIR}/")


if __name__ == "__main__":
    generate_diagnostics()