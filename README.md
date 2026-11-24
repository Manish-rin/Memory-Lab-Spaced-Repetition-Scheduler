# Memory-Lab-Spaced-Repetition-Scheduler
Adaptive Spaced-Repetition Scheduler built with Polars, scikit-learn, and Streamlit. Trained on 1.08M review logs from the MaiMemo open-source dataset (Ye et al., KDD 2022) using isotonic-calibrated, monotonic HistGradientBoosting to forecast forgetting curves and optimal review days.


# MemoryLab — Adaptive Spaced-Repetition Scheduler

An end-to-end Machine Learning pipeline and interactive **Streamlit** web application that predicts personalized Ebbinghaus forgetting curves and schedules vocabulary reviews right before a learner's recall probability drops below a target retention threshold.

---

## Dataset Source & Research Paper Foundation

This project is built on the open-source **MaiMemo Student Memory Logs** dataset (`maimemo/student-memory-logs` on Hugging Face / official MaiMemo open-source release), introduced in the ACM SIGKDD 2022 paper:

> **A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling**  
> Junyao Ye, Jingyong Su, Yilong Cao *(Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD '22)*

### Dataset Scale & Structure
- **Sampled & Cleaned Corpus:** `1,081,771` vocabulary review logs across `4,167` distinct students, joined with empirical lexical difficulty scores (`d`).
- **Target Variable (`r`):** Binary recall outcome (`1` = recalled `75.86%`, `0` = lapsed/forgotten `24.14%`).
- **Temporal History:** Sequence of prior review intervals (`t_history`), prior recall outcomes (`r_history`), repetition index (`i`), and elapsed days since last review (`delta_t`).

---

## What This Project Does & How It Works

Traditional flashcard applications rely on rigid, hand-tuned spacing rules (such as fixed `1 -> 3 -> 7` day intervals) regardless of word difficulty or past lapses. **MemoryLab** replaces static heuristics with data-driven probabilistic forecasting:

1. **High-Throughput Feature Engineering (`src/features.py` via Polars):**
   - Extracts **17 cognitive memory features** from raw comma-separated review histories, including total recall/lapse counts, historical lapse ratio, recent 3-attempt accuracy, last/mean/maximum intervals, log-transformed elapsed time (`log_delta_t`), elapsed-to-last interval ratios, and word character length.
2. **Leak-Free User Evaluation (`GroupShuffleSplit`):**
   - Splits data strictly by user ID (`844,860` training rows / `236,911` unseen-user test rows) so the model cannot memorize individual learner habits.
3. **Physics-Informed Monotonic Modeling (`src/train.py`):**
   - Trains three `scikit-learn` models (`LogisticRegression`, `RandomForestClassifier`, and `HistGradientBoostingClassifier`).
   - Enforces **monotonic constraints (`monotonic_cst = -1`)** on elapsed time (`delta_t`, `log_delta_t`, `elapsed_to_last_ratio`) and word difficulty (`d`) to eliminate observational selection bias and guarantee smooth Ebbinghaus decay over time.
4. **Isotonic Probability Calibration:**
   - Wraps the gradient boosted ensemble in `CalibratedClassifierCV(method="isotonic")` so predicted recall probabilities align directly with empirical retention frequencies.
5. **Interactive Scheduler Web App (`app.py`):**
   - **Curve & Review Planner:** Sweeps $\Delta t \in [1, 90]$ days to plot interactive Plotly forgetting curves, toggle past review outcomes live, compare counterfactual lapse curves, and pinpoint the optimal next review day.
   - **Flashcard Simulator:** Interactive vocabulary quiz that logs recall/lapse outcomes and computes next review intervals in real time.

---

## Benchmark Results (236,911 Unseen-User Test Rows)

| Model | ROC-AUC | Log-Loss | Brier Score | Accuracy | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.7008 | 0.5009 | 0.1637 | 76.55% | Linear Baseline |
| **Random Forest (100 Trees)** | 0.7070 | 0.4975 | 0.1624 | 76.74% | Bagging Ensemble |
| **HistGradientBoosting + Isotonic Calibration** | **0.7076** | **0.4971** | **0.1623** | **76.76%** | **Selected Production Model** |

---

## Project Structure

```text
spaced-repetition-scheduler/
├── .streamlit/config.toml        # Clean light theme configuration
├── data/
│   ├── raw/                      # Original MaiMemo TSV files (gitignored)
│   └── processed/                # Cleaned CSVs & engineered Parquet features
├── models/
│   └── syllabus_gb_model.joblib  # Serialized calibrated monotonic GBDT model
├── reports/                      # ROC, calibration, feature importance & forgetting plots
├── src/
│   ├── ingest.py                 # Chunked user-hashed sampling from raw TSV
│   ├── features.py               # Polars vectorized cognitive feature pipeline
│   ├── train.py                  # GroupShuffleSplit training & isotonic calibration
│   └── evaluate.py               # Diagnostic plots & metric verification
├── app.py                        # Interactive Streamlit dashboard
└── README.md



# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install polars pandas numpy scikit-learn joblib plotly matplotlib streamlit

# 3. Run feature engineering, training, and evaluation
python src/features.py
python src/train.py
python src/evaluate.py

# 4. Launch the interactive web application
streamlit run app.py





