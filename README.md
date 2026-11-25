# MemoryLab: Adaptive Spaced-Repetition Memory Scheduler

**MemoryLab** is an end-to-end Machine Learning system and interactive **Streamlit** web application that models personalized Ebbinghaus forgetting curves from empirical student review logs and calculates the optimal next review interval right before a learner's recall probability drops below a target retention threshold.

Rather than relying on static, rule-based flashcard spacing heuristics (such as fixed `1 -> 3 -> 7 -> 15` day schedules or SuperMemo SM-2 multipliers), MemoryLab predicts continuous recall probability $P(r_i = 1 \mid \mathbf{x}_i, \Delta t)$ across candidate future review delays $\Delta t \in [1, 90]$ days using **physics-informed, monotonically constrained, and isotonic-calibrated Gradient Boosted Decision Trees**.

---

## Real-World Context: How Adaptive Card-Based Recall Works

Modern language-learning and flashcard platforms such as **MaiMemo**, **Duolingo** (Half-Life Regression), and **Anki** (FSRS) all face the same core scheduling trade-off:

* **Review too early:** The learner wastes study time reviewing flashcards they already remember effortlessly.
* **Review too late:** The memory trace decays past the point of retrieval, forcing the learner to relearn the word from scratch.

### Concrete Flashcard Example

Consider a learner studying the English vocabulary card **`"ephemeral"`** (intrinsic lexical difficulty $d = 8.80$) across four prior sessions:

* **Review History:** `t_history = [0, 1, 3, 7]` days | `r_history = [0, 1, 1, 0]` *(Failed on day 0, recalled on days 1 & 3, lapsed after a 7-day gap)*
* **Target Retention Threshold ($\tau$):** `80%` ($\tau = 0.80$)

Instead of blindly applying a fixed multiplier after a lapse, MemoryLab evaluates the learner's calibrated forgetting curve $\hat{p}(\Delta t)$ across every future day $\Delta t \in \{1, 2, \dots, 90\}$:

| Candidate Delay ($\Delta t$) | Predicted Recall Probability $\hat{p}(\Delta t)$ | Meets $\tau \ge 0.80$? | Scheduling Decision |
| :---: | :---: | :---: | :--- |
| **1 day** | `88.4%` | Yes | Too early; retention is well above target |
| **3 days** | `83.1%` | Yes | Safe, but still leaves unused memory stability |
| **5 days** | **`80.2%`** | **Yes** | **Optimal Review Day ($\Delta t^* = 5$)** — maximizes spacing before drop-off |
| **6 days** | `78.9%` | No | Drops below the 80% target threshold |
| **14 days** | `64.5%` | No | High risk of forgetting |

By scheduling the card on **Day 5**, MemoryLab maximizes long-term memory consolidation per minute studied while respecting the learner's recent lapse history and the word's higher lexical difficulty.

---

## Dataset Source & Research Paper Foundation

This project is built on the open-source **MaiMemo Student Memory Logs** dataset (`maimemo/student-memory-logs` on Hugging Face / official MaiMemo open-source release), introduced in the ACM SIGKDD 2022 paper:

> **A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling**  
> Junyao Ye, Jingyong Su, Yilong Cao *(Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD '22)*

### How to Download the Dataset

1. **Create a Hugging Face Account & Accept Terms:**
   Visit the official dataset repository at [https://huggingface.co/datasets/Maimemo/student-memory-logs](https://huggingface.co/datasets/Maimemo/student-memory-logs), log in, and accept the **CC-BY-NC-4.0** dataset access agreement.
2. **Download the Required TSV Files (`opensource_dataset.tsv` and `opensource_dataset_difficulty.tsv`):**
   Using the `huggingface_hub` CLI:

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   mkdir -p data/raw
   huggingface-cli download Maimemo/student-memory-logs opensource_dataset.tsv --repo-type dataset --local-dir data/raw
   huggingface-cli download Maimemo/student-memory-logs opensource_dataset_difficulty.tsv --repo-type dataset --local-dir data/raw
---
### Dataset Scale & Structure
- **Sampled & Cleaned Corpus:** `1,081,771` vocabulary review logs across `4,167` distinct students, joined with empirical lexical difficulty scores (`d`).
- **Target Variable (`r`):** Binary recall outcome (`1` = recalled `75.86%`, `0` = lapsed/forgotten `24.14%`).
- **Temporal History:** Sequence of prior review intervals (`t_history`), prior recall outcomes (`r_history`), repetition index (`i`), and elapsed days since last review (`delta_t`).

---

## 1. Project Overview & Architecture

Human memory decays exponentially over time (the **Ebbinghaus Forgetting Curve**), while active retrieval practice slows that decay rate (the **Spacing Effect** and **Lag Effect**). Modeling these dynamics from real-world observational logs introduces two major ML challenges:
1. **Observational Selection Bias:** In naturalistic study apps, learners who wait 45+ days before reviewing a word tend to be advanced learners reviewing easy words, which causes unconstrained tree models to falsely predict *rising* recall over longer delays.
2. **Probability Calibration:** A spaced-repetition scheduler relies directly on predicted probabilities (e.g., *"schedule review on the day $P(\text{recall})$ crosses $0.80$"*). Uncalibrated tree ensembles distort probability magnitudes even when their ROC-AUC ranking is strong.

MemoryLab solves both problems through a five-stage pipeline:

```text
      Raw MaiMemo TSVs (15.5 GB / 220M+ logs)
                          │
                          ▼
       [src/ingest.py & src/data_cleaning_*.ipynb]
  User-Hashed Sample & Lexical Difficulty Join (1,081,771 rows, 4,167 users)
                          │
                          ▼
       [src/features.py — Vectorized Polars Pipeline]
17 Cognitive Temporal & Lexical Features (features_1m.parquet)
                          │
                          ▼ 
       [src/train.py — Leak-Free GroupShuffleSplit by User ID]
Monotonic HistGradientBoosting (monotonic_cst = -1) + Isotonic Calibration
                          │
                          ▼ 
            [src/evaluate.py & app.py]
 Diagnostic Reports (reports/) & Interactive Streamlit Scheduler Dashboard
```
## Dataset Schema

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `u` | `string` | Anonymized student identifier | `23af1d` |
| `w` | `string` | Target English vocabulary word | `ephemeral` |
| `i` | `int` | Current repetition index (1-based review count) | `5` |
| `t_history` | `string` | Comma-separated sequence of prior inter-review intervals (days) | `"0,1,3,7"` |
| `r_history` | `string` | Comma-separated sequence of prior binary recall outcomes (`1` = Pass, `0` = Fail) | `"0,1,1,0"` |
| `delta_t` | `int` | Elapsed days since the $(i-1)$-th review | `14` |
| `r` | `int` | **Target label:** Binary recall outcome at review $i$ (`1` = Recalled, `0` = Forgotten) | `1` |
| `d` | `float` | Empirical lexical difficulty score from `opensource_dataset_difficulty.tsv` | `8.80` |



## 4. Algorithms & Methodology
**A. Cognitive Feature Engineering (src/features.py)**

Raw variable-length comma-separated sequences (t_history, r_history) cannot be fed directly into tabular decision trees. We engineer 17 domain-specific cognitive memory features capturing repetition strength, spacing magnitude, recency, and lexical complexity:

| Category | Feature Name | Mathematical / Logical Definition | Cognitive Role |
|---|---|---|---|
| Time Decay | `delta_t` | Days elapsed since the last review | Primary Ebbinghaus decay variable |
| Time Decay | `log_delta_t` | $\ln(1+\Delta t)$ | Captures non-linear log-time forgetting rate |
| Time Decay | `elapsed_to_last_ratio` | $\frac{\Delta t}{\Delta t_{\text{last}}+1}$ | Measures schedule stretch relative to the prior interval |
| Lexical | `d` | Empirical word difficulty score | Baseline intrinsic item difficulty |
| Lexical | `word_len` | Number of characters in word $w$ | Orthographic complexity proxy |
| Repetition | `i` | Current repetition number | Total exposure count |
| Repetition | `history_len` | Length of the `r_history` sequence | Prior review depth |
| Retrieval | `total_recalls` | $\sum_{k=1}^{n} r_k$ | Cumulative successful retrieval strength |
| Retrieval | `total_lapses` | $\sum_{k=1}^{n}(1-r_k)$ | Cumulative forgetting history |
| Retrieval | `historical_accuracy` | $\frac{\text{total\_recalls}}{\text{history\_len}}$ | Learner-item mastery ratio |
| Retrieval | `lapse_ratio` | $\frac{\text{total\_lapses}}{\text{history\_len}}$ | Item-specific failure propensity |
| Recency | `last_outcome` | $r_{i-1} \in \{0,1\}$ | Immediate previous retrieval outcome |
| Recency | `recent_3_acc` | $\frac{1}{3}\sum_{k=i-3}^{i-1}r_k$ | Short-term retention momentum |
| Spacing | `last_interval` | Most recent interval, $\Delta t_{i-1}$ | Current memory stability proxy |
| Spacing | `mean_interval` | $\frac{1}{n}\sum_{k=1}^{n}\Delta t_k$ | Average historical spacing |
| Spacing | `max_interval` | $\max(\Delta t_1,\Delta t_2,\ldots,\Delta t_n)$ | Longest successfully bridged gap |
| Spacing | `total_study_span` | $\sum_{k=1}^{n}\Delta t_k$ | Total cumulative age of the memory trace |


### B. Leak-Free User-Grouped Validation

A critical pitfall in educational data mining is random row-level splitting, which places different reviews from the same student in both train and test sets allowing models to overfit individual user IDs rather than generalizing memory physics. We enforce `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped by student ID (`u`), guaranteeing **100% unseen students** in the 236,911-row evaluation set.

### C. Monotonic Gradient Boosted Decision Trees (`monotonic_cst = -1`)

We train `HistGradientBoostingClassifier` with explicit negative monotonic constraints (`monotonic_cst = -1`) on all time-decay and difficulty features (`delta_t`, `log_delta_t`, `elapsed_to_last_ratio`, and `d`), while leaving history-strength features unconstrained (`0`).

During tree split construction, `monotonic_cst = -1` enforces:

$$x_j^{(a)} > x_j^{(b)} \implies \hat{f}(\mathbf{x}^{(a)}) \le \hat{f}(\mathbf{x}^{(b)})$$

for every constrained feature $j$. This guarantees that:

1. For any fixed word and review history, predicted recall probability **never increases** as elapsed days $\Delta t$ grow.
2. Higher word difficulty $d$ **never yields a higher recall probability** than an easier word under identical conditions.

### D. Isotonic Probability Calibration & Optimal Scheduling Policy

To ensure predicted probabilities reflect true empirical retention frequencies, we wrap the monotonic `HistGradientBoostingClassifier` inside `CalibratedClassifierCV(method="isotonic", cv=3)`, fitting a non-parametric piecewise-constant monotonic step function via Pool Adjacent Violators Algorithm (PAVA) on out-of-fold validation predictions.

At inference time, given a learner's target retention threshold $\tau \in [0.50, 0.99]$ (default $\tau = 0.80$), the scheduler evaluates $\hat{p}(\Delta t) = \hat{P}(r = 1 \mid \mathbf{x}, \Delta t)$ across $\Delta t \in \{1, 2, \dots, 90\}$ and selects the optimal next review day:

$$\Delta t^* = \max \left\{ \Delta t \in [1, 90] \;\middle|\; \hat{p}(\Delta t) \ge \tau \right\}$$


## 5. Experimental Results & Performance Comparison

All models were trained on the exact same 844,860-row user-grouped training partition and evaluated on 236,911 unseen-user test instances.

### Quantitative Model Benchmark

| Model Architecture | ROC-AUC (Higher is Better) | Log-Loss (Lower is Better) | Brier Score (Lower is Better) | Accuracy (Threshold 0.5) | Engineering & Deployment Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** (`StandardScaler` + L2) | 0.7008 | 0.5009 | 0.1637 | 76.55% | Linear parametric baseline |
| **Random Forest** (100 Trees, `max_depth=16`) | 0.7070 | 0.4975 | 0.1624 | 76.74% | Non-linear bagging ensemble |
| **HistGradientBoosting + Monotonic Constraints + Isotonic Calibration** | **0.7076** | **0.4971** | **0.1623** | **76.76%** | **Selected Production Scheduler** |

## 6. Instructions for Executing the Project

### Prerequisites
* **Python:** 3.10 or higher (tested on Python 3.14 on Ubuntu Linux)
* **RAM:** 8 GB+ recommended if running full feature engineering on 1.08M rows (`app.py` inference requires < 500 MB RAM)

### Step 1: Clone the Repository & Set Up Virtual Environment

```bash
git clone [https://github.com/Manish-rin/Memory-Lab-Spaced-Repetition-Scheduler.git](https://github.com/Manish-rin/Memory-Lab-Spaced-Repetition-Scheduler.git)
cd Memory-Lab-Spaced-Repetition-Scheduler

python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash

pip install --upgrade pip
pip install polars pandas numpy scikit-learn joblib plotly matplotlib shap streamlit
```
### License & Acknowledgments

    Dataset: MaiMemo Inc. (Maimemo/student-memory-logs), licensed under CC-BY-NC-4.0.

    Research Foundation: Ye, J., Su, J., & Cao, Y. (2022). A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling. KDD '22.
