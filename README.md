# MemoryLab: Adaptive Spaced-Repetition Memory Scheduler

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/app-Streamlit-ff4b4b)
![Dataset License](https://img.shields.io/badge/dataset-CC--BY--NC--4.0-lightgrey)

**MemoryLab** is an end-to-end machine learning system and interactive **Streamlit** app that models personalized Ebbinghaus forgetting curves from real student review logs. It then picks the next review interval right before predicted recall drops below a target retention threshold.

Instead of fixed schedules (`1 -> 3 -> 7 -> 15` days) or SM-2 style multipliers, MemoryLab predicts the recall probability $P(r_i = 1 \mid \mathbf{x}_i, \Delta t)$ for every candidate delay $\Delta t \in [1, 90]$ days, using **monotonically constrained, isotonic-calibrated Gradient Boosted Decision Trees**.

**TL;DR**

- Input: a learner's review history for a word, plus a target retention (default 80%).
- Output: the latest day on which predicted recall is still at or above the target.
- Key idea: a monotonic constraint forces recall to never increase with elapsed time, and isotonic calibration makes the probabilities trustworthy.

<!-- TODO: add a screenshot or GIF of the Streamlit app here:
![MemoryLab demo](reports/demo.gif) -->

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How Adaptive Card-Based Recall Works](#how-adaptive-card-based-recall-works)
3. [Dataset](#dataset)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Limitations](#limitations)
8. [Project Structure](#project-structure)
9. [License & Citation](#license--citation)

---

## Quick Start

**Prerequisites:** Python 3.10+ (tested on Python 3.14, Ubuntu Linux). 8 GB+ RAM is recommended for full feature engineering on 1.08M rows; the app itself needs under 500 MB.

```bash
# 1. Clone and set up a virtual environment
git clone https://github.com/Manish-rin/Memory-Lab-Spaced-Repetition-Scheduler.git
cd Memory-Lab-Spaced-Repetition-Scheduler

python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install polars pandas numpy scikit-learn joblib plotly matplotlib shap streamlit

# 3. Download the dataset (see the Dataset section below)

# 4. Run the pipeline
python src/ingest.py      # sample users + join lexical difficulty
python src/features.py    # build the 17 features -> features_1m.parquet
python src/train.py       # train monotonic HGB + isotonic calibration
python src/evaluate.py    # write diagnostic reports to reports/

# 5. Launch the scheduler dashboard
streamlit run app.py
```

> If you only want to try the app, you can skip steps 3-4 only if trained model artifacts are already present in the repo.

---

## How Adaptive Card-Based Recall Works

Platforms such as **MaiMemo**, **Duolingo** (Half-Life Regression) and **Anki** (FSRS) all face the same trade-off:

- **Review too early:** the learner wastes time on cards they already remember.
- **Review too late:** the memory has decayed, and the learner must relearn the word.

### Illustrative Example

> **Note:** the probabilities below are **illustrative** and show how the scheduler makes a decision. They are not reproducible model output.

A learner studies the card **`"ephemeral"`** (lexical difficulty $d = 8.80$):

- **Review history:** `t_history = [0, 1, 3, 7]` days, `r_history = [0, 1, 1, 0]` (failed on day 0, recalled on days 1 and 3, lapsed after a 7-day gap)
- **Target retention:** $\tau = 0.80$

| Candidate Delay ($\Delta t$) | Predicted Recall $\hat{p}(\Delta t)$ | Meets $\tau \ge 0.80$? | Decision |
| :---: | :---: | :---: | :--- |
| 1 day | 88.4% | Yes | Too early; retention far above target |
| 3 days | 83.1% | Yes | Safe, but wastes some memory stability |
| **5 days** | **80.2%** | **Yes** | **Optimal ($\Delta t^* = 5$)** |
| 6 days | 78.9% | No | Falls below target |
| 14 days | 64.5% | No | High risk of forgetting |

---

## Dataset

MemoryLab uses the open-source **MaiMemo Student Memory Logs** dataset, introduced in:

> Ye, J., Su, J., & Cao, Y. (2022). *A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling.* Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22).

### Download

1. Create a Hugging Face account and accept the **CC-BY-NC-4.0** terms at <https://huggingface.co/datasets/Maimemo/student-memory-logs>.
2. Download the two required files:

```bash
pip install huggingface_hub
huggingface-cli login
mkdir -p data/raw
huggingface-cli download Maimemo/student-memory-logs opensource_dataset.tsv --repo-type dataset --local-dir data/raw
huggingface-cli download Maimemo/student-memory-logs opensource_dataset_difficulty.tsv --repo-type dataset --local-dir data/raw
```

### Scale

- **Raw data:** about 15.5 GB, 220M+ logs.
- **Sampled and cleaned corpus:** `1,081,771` review logs from `4,167` students, joined with lexical difficulty scores (`d`).
- **Target (`r`):** `1` = recalled (75.86%), `0` = lapsed (24.14%).

### Schema

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `u` | string | Anonymized student ID | `23af1d` |
| `w` | string | Target English word | `ephemeral` |
| `i` | int | Current repetition index (1-based review count) | `5` |
| `t_history` | string | Comma-separated prior inter-review intervals (days) | `"0,1,3,7"` |
| `r_history` | string | Comma-separated prior recall outcomes (1 = pass, 0 = fail) | `"0,1,1,0"` |
| `delta_t` | int | Days elapsed since the $(i-1)$-th review | `14` |
| `r` | int | **Target:** recall outcome at review $i$ | `1` |
| `d` | float | Lexical difficulty from `opensource_dataset_difficulty.tsv` | `8.80` |

---

## Pipeline Architecture

Human memory decays roughly exponentially over time (the **Ebbinghaus forgetting curve**), and active retrieval slows that decay (the **spacing effect**). Learning this from observational logs raises two problems:

1. **Observational selection bias.** Learners who wait 45+ days to review a word are often advanced learners reviewing easy words. Unconstrained tree models can therefore learn that recall *rises* with longer delays. MemoryLab **mitigates** this by enforcing monotonic decay in the model.
2. **Probability calibration.** The scheduler thresholds predicted probabilities directly, so they must match real retention frequencies, not just rank well (high AUC).

```text
      Raw MaiMemo TSVs (15.5 GB / 220M+ logs)
                          │
                          ▼
              [src/ingest.py, src/data_cleaning_*.ipynb]
   User-hashed sample + lexical difficulty join (1,081,771 rows, 4,167 users)
                          │
                          ▼
              [src/features.py — vectorized Polars]
        17 cognitive temporal & lexical features (features_1m.parquet)
                          │
                          ▼
              [src/train.py — GroupShuffleSplit by user ID]
   Monotonic HistGradientBoosting (monotonic_cst = -1) + isotonic calibration
                          │
                          ▼
                  [src/evaluate.py, app.py]
      Diagnostic reports (reports/) + Streamlit scheduler dashboard
```

---

## Methodology

### A. Feature Engineering (`src/features.py`)

Variable-length comma-separated histories cannot go straight into a tabular tree model, so they are converted into 17 features:

| Category | Feature | Definition | Role |
| :--- | :--- | :--- | :--- |
| Time decay | `delta_t` | Days since last review | Primary decay variable |
| Time decay | `log_delta_t` | $\ln(1+\Delta t)$ | Non-linear forgetting rate |
| Time decay | `elapsed_to_last_ratio` | $\frac{\Delta t}{\Delta t_{\text{last}}+1}$ | Schedule stretch vs. previous interval |
| Lexical | `d` | Empirical word difficulty | Intrinsic item difficulty |
| Lexical | `word_len` | Characters in word $w$ | Orthographic complexity proxy |
| Repetition | `i` | Current repetition number | Total exposure count |
| Repetition | `history_len` | Length of `r_history` | Prior review depth |
| Retrieval | `total_recalls` | $\sum_{k=1}^{n} r_k$ | Cumulative retrieval strength |
| Retrieval | `total_lapses` | $\sum_{k=1}^{n}(1-r_k)$ | Cumulative forgetting |
| Retrieval | `historical_accuracy` | $\text{total\_recalls} / \text{history\_len}$ | Learner-item mastery ratio |
| Retrieval | `lapse_ratio` | $\text{total\_lapses} / \text{history\_len}$ | Item-specific failure rate |
| Recency | `last_outcome` | $r_{i-1} \in \{0,1\}$ | Most recent outcome |
| Recency | `recent_3_acc` | $\frac{1}{3}\sum_{k=i-3}^{i-1} r_k$ | Short-term momentum |
| Spacing | `last_interval` | $\Delta t_{i-1}$ | Memory stability proxy |
| Spacing | `mean_interval` | $\frac{1}{n}\sum_{k=1}^{n} \Delta t_k$ | Average spacing |
| Spacing | `max_interval` | $\max(\Delta t_1,\ldots,\Delta t_n)$ | Longest bridged gap |
| Spacing | `total_study_span` | $\sum_{k=1}^{n} \Delta t_k$ | Age of the memory trace |

### B. Leak-Free, User-Grouped Validation

Random row-level splits put reviews from the same student in both train and test, so models memorize users instead of learning memory dynamics. MemoryLab uses `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped by student ID (`u`). The 236,911-row test set contains **only unseen students**.

### C. Monotonic Gradient Boosted Trees

`HistGradientBoostingClassifier` is trained with `monotonic_cst = -1` on `delta_t`, `log_delta_t`, `elapsed_to_last_ratio` and `d`. History-strength features are left unconstrained (`0`). For each constrained feature $j$:

$$x_j^{(a)} > x_j^{(b)} \implies \hat{f}(\mathbf{x}^{(a)}) \le \hat{f}(\mathbf{x}^{(b)})$$

This guarantees that, for a fixed word and history:

1. Predicted recall **never increases** as $\Delta t$ grows.
2. A harder word (higher $d$) **never gets a higher recall probability** than an easier one under identical conditions.

### D. Isotonic Calibration and Scheduling Policy

The classifier is wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)`, which fits a monotonic step function (via the Pool Adjacent Violators Algorithm) on out-of-fold predictions.

Given a target retention $\tau \in [0.50, 0.99]$ (default $0.80$), the scheduler evaluates $\hat{p}(\Delta t)$ for $\Delta t \in \{1, \dots, 90\}$ and selects:

$$\Delta t^* = \max \left\{ \Delta t \in [1, 90] \;\middle|\; \hat{p}(\Delta t) \ge \tau \right\}$$

Because $\hat{p}$ is monotonically non-increasing in $\Delta t$, this is the last day before predicted recall falls below $\tau$.

---

## Results

All models were trained on the same 844,860-row user-grouped training set and evaluated on 236,911 rows from unseen students.

| Model | ROC-AUC ↑ | Log-Loss ↓ | Brier ↓ | Accuracy (0.5) | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Logistic Regression (`StandardScaler` + L2) | 0.7008 | 0.5009 | 0.1637 | 76.55% | Linear baseline |
| Random Forest (100 trees, `max_depth=16`) | 0.7070 | 0.4975 | 0.1624 | 76.74% | Non-linear ensemble |
| **HistGB + Monotonic + Isotonic** | **0.7076** | **0.4971** | **0.1623** | **76.76%** | **Selected scheduler** |

**Takeaway:** the monotonic, calibrated model matches or slightly beats the unconstrained models on accuracy metrics while guaranteeing physically sensible forgetting curves. The main benefit is trustworthy, monotone probabilities, not a large jump in AUC.

Diagnostic plots (calibration curves, SHAP, etc.) are written to `reports/`.

<!-- TODO: add reliability diagram + ECE here once computed, e.g.
![Calibration curve](reports/calibration.png) -->

---

## Limitations

- **Observational data.** The monotonic constraint imposes a prior and mitigates, but does not remove, selection bias in the logs.
- **Sampled data.** Training uses about 1.08M rows from 4,167 users out of 220M+ logs.
- **Calibration folds.** `CalibratedClassifierCV(cv=3)` with an integer `cv` uses standard K-fold, so the same student can appear in both the fit and calibration folds. Passing a `GroupKFold` split grouped by `u` would remove this.
- **Cold start.** There is no per-user personalization for brand-new learners with little history.
- **No heuristic baselines yet.** SM-2, fixed schedules and Half-Life Regression are not benchmarked here, so this README makes no claim of beating them.
- **Commercial use.** The dataset is CC-BY-NC-4.0, which restricts commercial use.

---

## Project Structure

```text
.
├── app.py                  # Streamlit scheduler dashboard
├── src/
│   ├── ingest.py           # user-hashed sampling + difficulty join
│   ├── data_cleaning_*.ipynb
│   ├── features.py         # 17-feature Polars pipeline
│   ├── train.py            # monotonic HGB + isotonic calibration
│   └── evaluate.py         # metrics and diagnostic reports
├── data/
│   └── raw/                # downloaded TSVs (not committed)
├── reports/                # evaluation outputs and plots
└── README.md
```

---

## License & Citation

- **Dataset:** MaiMemo Inc., [`Maimemo/student-memory-logs`](https://huggingface.co/datasets/Maimemo/student-memory-logs), licensed under **CC-BY-NC-4.0**.
- **Code:** add a `LICENSE` file to the repository and reference it here.

```bibtex
@inproceedings{ye2022stochastic,
  title     = {A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling},
  author    = {Ye, Junyao and Su, Jingyong and Cao, Yilong},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2022}
}
```
