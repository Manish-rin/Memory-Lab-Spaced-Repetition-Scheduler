import math
from pathlib import Path
from typing import List
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "syllabus_gb_model.joblib"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="MemoryLab Inference Engine")
loaded_artifact = joblib.load(MODEL_PATH)

if isinstance(loaded_artifact, dict):
    model = next(
        (v for v in loaded_artifact.values() if hasattr(v, "predict_proba")),
        None,
    )
    EXPECTED_COLS = list(
        loaded_artifact.get("features")
        or loaded_artifact.get("feature_names")
        or getattr(model, "feature_names_in_", [])
    )
else:
    model = loaded_artifact
    EXPECTED_COLS = list(getattr(model, "feature_names_in_", []))

print("Loaded model features:", EXPECTED_COLS)


class CurveRequest(BaseModel):
    word: str = "ephemeral"
    d: float = 8.8
    t_history: List[int] = [0, 1, 3, 7]
    r_history: List[int] = [1, 1, 1, 0]
    target_retention: float = 0.80
    forecast_window: int = 60
    include_counterfactual: bool = True


class FlashcardStepRequest(BaseModel):
    word: str
    d: float
    t_history: List[int]
    r_history: List[int]
    elapsed_days: int = 1
    outcome: int = 1
    target_retention: float = 0.80


def build_feature_df(word: str, d: float, t_hist: List[int], r_hist: List[int], window: int) -> pd.DataFrame:
    word_len = max(1, len(word.strip()))
    hist_len = len(r_hist)
    i_val = hist_len + 1
    recall_count = sum(r_hist)
    lapse_count = sum(1 - r for r in r_hist)
    historical_accuracy = recall_count / max(1, hist_len)
    lapse_ratio = lapse_count / max(1, hist_len)
    last_outcome = r_hist[-1] if hist_len > 0 else 1
    recent_3_recall_rate = (sum(r_hist[-3:]) / min(3, hist_len)) if hist_len > 0 else 1.0

    last_interval = float(t_hist[-1]) if len(t_hist) > 0 else 0.0
    mean_interval = (sum(t_hist) / len(t_hist)) if len(t_hist) > 0 else 0.0
    max_interval = float(max(t_hist)) if len(t_hist) > 0 else 0.0
    total_study_span = float(sum(t_hist))
    log_last_interval = math.log1p(last_interval)

    rows = []
    for day in range(1, window + 1):
        feat_map = {
            "d": float(d),
            "word_len": word_len,
            "i": i_val,
            "hist_len": hist_len,
            "history_len": hist_len,
            "recall_count": recall_count,
            "total_recalls": recall_count,
            "lapse_count": lapse_count,
            "total_lapses": lapse_count,
            "historical_accuracy": historical_accuracy,
            "lapse_ratio": lapse_ratio,
            "last_outcome": last_outcome,
            "recent_3_recall_rate": recent_3_recall_rate,
            "recent_3_acc": recent_3_recall_rate,
            "last_interval": last_interval,
            "log_last_interval": log_last_interval,
            "mean_interval": mean_interval,
            "max_interval": max_interval,
            "total_study_span": total_study_span,
            "delta_t": day,
            "log_delta_t": math.log1p(day),
            "elapsed_to_last_ratio": day / (last_interval + 1.0),
            "elapsed_to_max_ratio": day / (max_interval + 1.0),
        }
        rows.append(feat_map)

    df = pd.DataFrame(rows)
    if EXPECTED_COLS:
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df = df[EXPECTED_COLS]
    return df


def run_inference(word: str, d: float, t_hist: List[int], r_hist: List[int], target: float, window: int, include_cf: bool):
    df = build_feature_df(word, d, t_hist, r_hist, window)
    probs = model.predict_proba(df)[:, 1].tolist()
    points = [{"day": idx + 1, "prob": round(float(p), 5)} for idx, p in enumerate(probs)]

    cf_points = []
    if include_cf and len(r_hist) > 0:
        r_cf = list(r_hist)
        r_cf[-1] = 0 if r_cf[-1] == 1 else 1
        df_cf = build_feature_df(word, d, t_hist, r_cf, window)
        cf_probs = model.predict_proba(df_cf)[:, 1].tolist()
        cf_points = [{"day": idx + 1, "prob": round(float(p), 5)} for idx, p in enumerate(cf_probs)]

    optimal_day = 1
    for pt in points:
        if pt["prob"] >= target:
            optimal_day = pt["day"]
        else:
            break

    p_24h = points[0]["prob"]
    p_7d = points[min(6, len(points) - 1)]["prob"]
    p_30d = points[min(29, len(points) - 1)]["prob"]

    return {
        "points": points,
        "cf_points": cf_points,
        "optimal_day": optimal_day,
        "kpi": {"p_24h": p_24h, "p_7d": p_7d, "p_30d": p_30d},
        "features_day1": df.iloc[0].to_dict(),
    }


@app.post("/api/predict-curve")
def predict_curve(req: CurveRequest):
    return run_inference(
        req.word, req.d, req.t_history, req.r_history,
        req.target_retention, req.forecast_window, req.include_counterfactual
    )


@app.post("/api/flashcard-step")
def flashcard_step(req: FlashcardStepRequest):
    new_t = list(req.t_history) + [req.elapsed_days]
    new_r = list(req.r_history) + [req.outcome]
    res = run_inference(req.word, req.d, new_t, new_r, req.target_retention, 60, False)
    return {
        "t_history": new_t,
        "r_history": new_r,
        "optimal_day": res["optimal_day"],
        "p_24h": res["kpi"]["p_24h"],
    }


if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


@app.get("/")
def serve_home():
    return FileResponse(STATIC_DIR / "index.html")
