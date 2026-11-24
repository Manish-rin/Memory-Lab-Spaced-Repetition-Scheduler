import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models/syllabus_gb_model.joblib"

st.set_page_config(
    page_title="MemoryLab · Spaced-Repetition Scheduler",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 100% !important;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif;
    }
    .hero-box {
        background: linear-gradient(120deg, #F8FAFC 0%, #EEF4FF 100%);
        border: 1px solid #DCE6F7;
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 18px;
    }
    .hero-tag {
        color: #0071E3;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .hero-heading {
        color: #0F172A;
        font-size: 25px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
    }
    .hero-text {
        color: #475569;
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or not {"model", "features"}.issubset(bundle):
        raise ValueError("The model file does not contain the expected model and feature list.")
    if not bundle["features"]:
        raise ValueError("The saved model has an empty feature list.")
    return bundle


def build_feature_row(d: float, t_list: list[float], r_list: list[int], delta_t: int, word: str) -> dict:
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
        "word_len": len(word.strip()),
    }


def predict_curve(cal_model, feature_cols, word: str, d: float, t_list: list[float], r_list: list[int], max_days: int):
    days = np.arange(1, max_days + 1)
    df_sweep = pd.DataFrame([build_feature_row(d, t_list, r_list, int(dt), word) for dt in days])[feature_cols]
    raw_probs = cal_model.predict_proba(df_sweep.to_numpy())[:, 1]
    smooth_probs = pd.Series(raw_probs).rolling(3, min_periods=1, center=True).mean().to_numpy()
    return days, raw_probs, smooth_probs


def init_state():
    if "word" not in st.session_state:
        st.session_state.word = "ephemeral"
        st.session_state.difficulty = 8.8
        st.session_state.t_list = [0.0, 1.0, 3.0, 7.0]
        st.session_state.r_list = [1, 1, 1, 1]
        st.session_state.preset = "Hard Word"


def set_preset(name: str):
    presets = {
        "Hard Word": ("ephemeral", 8.8, [0.0, 1.0, 3.0, 7.0], [1, 1, 1, 1]),
        "Easy Word": ("apple", 1.5, [0.0, 1.0, 3.0, 7.0], [1, 1, 1, 1]),
        "Recent Lapse": ("abandon", 5.5, [0.0, 1.0, 3.0, 7.0], [1, 1, 1, 0]),
        "New Word": ("civilian", 5.0, [0.0], [1]),
    }
    w, d, t_l, r_l = presets[name]
    st.session_state.word = w
    st.session_state.difficulty = d
    st.session_state.t_list = list(t_l)
    st.session_state.r_list = list(r_l)
    st.session_state.preset = name


def main():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at `{MODEL_PATH}`. Run `src/train.py` first.")
        st.stop()

    init_state()
    try:
        bundle = load_model()
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        st.error(
            "I couldn't load the saved model. Check that `models/syllabus_gb_model.joblib` "
            "is a valid model bundle, then run `src/train.py` to rebuild it."
        )
        st.caption(f"Details: {exc}")
        st.stop()
    cal_model = bundle["model"]
    feature_cols = bundle["features"]

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-tag">Cognitive Spaced-Repetition Engine · Calibrated Gradient Boosting</div>
            <div class="hero-heading">MemoryLab — Adaptive Spaced-Repetition Scheduler</div>
            <div class="hero-text">
                Trained on 1.08 million student review logs from MaiMemo, MemoryLab models individual memory decay curves and schedules vocabulary reviews right before recall drops below your target retention threshold.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Curve & Review Planner",
            "Flashcard Simulator",
            "Model Benchmark",
            "How It Works",
        ]
    )

  
    with tab1:
        left_col, right_col = st.columns([1.0, 2.6], gap="large")

        with left_col:
            with st.container(border=True):
                st.subheader("Parameters")

                p_c1, p_c2 = st.columns(2)
                if p_c1.button("Hard Word", width="stretch", type="primary" if st.session_state.preset == "Hard Word" else "secondary"):
                    set_preset("Hard Word")
                    st.rerun()
                if p_c2.button("Easy Word", width="stretch", type="primary" if st.session_state.preset == "Easy Word" else "secondary"):
                    set_preset("Easy Word")
                    st.rerun()

                p_c3, p_c4 = st.columns(2)
                if p_c3.button("Recent Lapse", width="stretch", type="primary" if st.session_state.preset == "Recent Lapse" else "secondary"):
                    set_preset("Recent Lapse")
                    st.rerun()
                if p_c4.button("New Word", width="stretch", type="primary" if st.session_state.preset == "New Word" else "secondary"):
                    set_preset("New Word")
                    st.rerun()

                st.divider()
                st.session_state.word = st.text_input("Vocabulary Word", value=st.session_state.word)
                st.session_state.difficulty = st.slider("Word Difficulty (1 = Easy, 10 = Hard)", 1.0, 10.0, float(st.session_state.difficulty), 0.1)
                target_retention = st.slider("Target Retention Goal", 0.70, 0.95, 0.85, 0.01, format="%.2f")
                max_days = st.select_slider("Forecast Window (Days)", options=[30, 45, 60, 90], value=60)
                show_counterfactual = st.toggle("Overlay Counterfactual Lapse Curve", value=True)

            with st.container(border=True):
                st.markdown("**Review History (Click to Toggle Outcome)**")
                cols = st.columns(min(len(st.session_state.r_list), 4))
                for idx, (iv, out) in enumerate(zip(st.session_state.t_list, st.session_state.r_list)):
                    c = cols[idx % len(cols)]
                    lbl = f"#{idx+1} ({int(iv)}d): {'Pass' if out == 1 else 'Fail'}"
                    if c.button(lbl, key=f"rev_{idx}", type="primary" if out == 1 else "secondary", width="stretch"):
                        st.session_state.r_list[idx] = 1 - st.session_state.r_list[idx]
                        st.rerun()

                b_add, b_rem = st.columns(2)
                if b_add.button("Add Recall", width="stretch"):
                    st.session_state.t_list.append(max(1.0, round(st.session_state.t_list[-1] * 2.0 + 1.0)))
                    st.session_state.r_list.append(1)
                    st.rerun()
                if b_rem.button("Remove Last", width="stretch", disabled=len(st.session_state.r_list) <= 1):
                    st.session_state.t_list.pop()
                    st.session_state.r_list.pop()
                    st.rerun()

        with right_col:
            days, raw_probs, smooth_probs = predict_curve(
                cal_model,
                feature_cols,
                st.session_state.word,
                st.session_state.difficulty,
                st.session_state.t_list,
                st.session_state.r_list,
                max_days,
            )
            safe_days = days[raw_probs >= target_retention]
            meets_target = len(safe_days) > 0
            optimal_day = int(safe_days[-1]) if meets_target else 1
            schedule_label = f"Day {optimal_day}" if meets_target else "Review now"
            schedule_context = (
                f"At or above {int(target_retention * 100)}% through day {optimal_day}"
                if meets_target and optimal_day == max_days
                else f"Target: {int(target_retention * 100)}%"
                if meets_target
                else "Recall is already below your target"
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Suggested Next Review", schedule_label, schedule_context)
            m2.metric("24-Hour Recall", f"{raw_probs[0] * 100:.1f}%", "1 day elapsed")
            m3.metric("7-Day Recall", f"{raw_probs[min(6, len(raw_probs)-1)] * 100:.1f}%", "1 week elapsed")
            m4.metric("30-Day Recall", f"{raw_probs[min(29, len(raw_probs)-1)] * 100:.1f}%", "1 month elapsed")

            with st.container(border=True):
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=smooth_probs * 100,
                        mode="lines",
                        name="Predicted Memory Retention",
                        line=dict(color="#0071E3", width=3.5, shape="spline", smoothing=0.6),
                        fill="tozeroy",
                        fillcolor="rgba(0, 113, 227, 0.08)",
                        hovertemplate="<b>Day %{x}</b><br>Recall Probability: <b>%{y:.1f}%</b><extra></extra>",
                    )
                )

                if show_counterfactual and st.session_state.r_list[-1] == 1:
                    _, _, lapse_smooth = predict_curve(
                        cal_model,
                        feature_cols,
                        st.session_state.word,
                        st.session_state.difficulty,
                        st.session_state.t_list,
                        st.session_state.r_list[:-1] + [0],
                        max_days,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=days,
                            y=lapse_smooth * 100,
                            mode="lines",
                            name="Counterfactual: If Last Review Failed",
                            line=dict(color="#EF4444", width=2, dash="dash", shape="spline", smoothing=0.6),
                            hovertemplate="<b>Day %{x} (After Lapse)</b><br>Recall: <b>%{y:.1f}%</b><extra></extra>",
                        )
                    )

                fig.add_hline(
                    y=target_retention * 100,
                    line_dash="dot",
                    line_color="#F59E0B",
                    line_width=2,
                    annotation_text=f"Target Threshold ({int(target_retention * 100)}%)",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#B45309", size=12),
                )

                opt_y = smooth_probs[optimal_day - 1] * 100
                fig.add_trace(
                    go.Scatter(
                        x=[optimal_day],
                        y=[opt_y],
                        mode="markers+text",
                        name=f"Optimal Review (Day {optimal_day})",
                        marker=dict(color="#10B981", size=13, line=dict(color="#FFFFFF", width=2.5)),
                        text=[f"  Review Day {optimal_day}"],
                        textposition="top right",
                        textfont=dict(color="#047857", size=12.5),
                        hoverinfo="skip",
                    )
                )

                fig.update_layout(
                    title=dict(
                        text=f'Ebbinghaus Forgetting Curve for "{st.session_state.word}" (Difficulty: {st.session_state.difficulty:.1f}/10)',
                        font=dict(size=16, color="#0F172A"),
                        x=0.01,
                    ),
                    xaxis=dict(
                        title="Days Elapsed Since Last Review",
                        range=[1, max_days],
                        showgrid=False,
                        zeroline=False,
                        linecolor="#CBD5E1",
                        tickfont=dict(color="#475569", size=12),
                        title_font=dict(color="#334155", size=13),
                    ),
                    yaxis=dict(
                        title="Predicted Recall Probability (%)",
                        range=[40, 103],
                        gridcolor="#F1F5F9",
                        zeroline=False,
                        linecolor="#CBD5E1",
                        tickfont=dict(color="#475569", size=12),
                        title_font=dict(color="#334155", size=13),
                    ),
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    hovermode="x unified",
                    height=500,
                    margin=dict(l=20, r=20, t=55, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(color="#334155", size=12),
                    ),
                )
                st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


    with tab2:
        deck = [
            {"word": "ephemeral", "meaning": "Lasting for a very short time", "d": 8.8},
            {"word": "ubiquitous", "meaning": "Present, appearing, or found everywhere", "d": 7.6},
            {"word": "pragmatic", "meaning": "Dealing with things sensibly and realistically", "d": 6.2},
            {"word": "ameliorate", "meaning": "Make something bad or unsatisfactory better", "d": 9.1},
            {"word": "candid", "meaning": "Truthful and straightforward; frank", "d": 4.3},
        ]
        if "quiz_idx" not in st.session_state:
            st.session_state.quiz_idx = 0
            st.session_state.show_meaning = False
            st.session_state.quiz_log = []

        card = deck[st.session_state.quiz_idx % len(deck)]
        _, center_card, _ = st.columns([1, 2, 1])
        with center_card:
            with st.container(border=True):
                st.caption(f"CARD {st.session_state.quiz_idx % len(deck) + 1} OF {len(deck)} · DIFFICULTY {card['d']}/10")
                st.markdown(f"## {card['word']}")
                if st.session_state.show_meaning:
                    st.info(card["meaning"])
                else:
                    st.write("Test your recall before revealing the definition.")

                c1, c2, c3 = st.columns(3)
                if c1.button("Reveal Definition", width="stretch"):
                    st.session_state.show_meaning = True
                    st.rerun()
                if c2.button("Remembered", type="primary", width="stretch"):
                    _, q_raw, _ = predict_curve(cal_model, feature_cols, card["word"], card["d"], [0.0, 1.0, 3.0], [1, 1, 1], 60)
                    safe = np.arange(1, 61)[q_raw >= target_retention]
                    opt_d = int(safe[-1]) if len(safe) > 0 else 1
                    st.session_state.quiz_log.insert(0, {"Word": card["word"], "Result": "Recalled", "Next Review": f"In {opt_d} days"})
                    st.session_state.quiz_idx += 1
                    st.session_state.show_meaning = False
                    st.rerun()
                if c3.button("Forgot", width="stretch"):
                    _, q_raw, _ = predict_curve(cal_model, feature_cols, card["word"], card["d"], [0.0, 1.0, 3.0], [1, 1, 0], 60)
                    safe = np.arange(1, 61)[q_raw >= target_retention]
                    opt_d = int(safe[-1]) if len(safe) > 0 else 1
                    st.session_state.quiz_log.insert(0, {"Word": card["word"], "Result": "Forgot", "Next Review": f"In {opt_d} days"})
                    st.session_state.quiz_idx += 1
                    st.session_state.show_meaning = False
                    st.rerun()

            if st.session_state.quiz_log:
                st.markdown("#### Session Scheduling Log")
                st.dataframe(pd.DataFrame(st.session_state.quiz_log), width="stretch", hide_index=True)


    with tab3:
        with st.container(border=True):
            st.subheader("Algorithm Benchmark (236,911 Unseen-User Test Rows)")
            bench_df = pd.DataFrame(
                [
                    {"Algorithm": "1. Logistic Regression", "ROC-AUC": 0.7008, "Log-Loss": 0.5009, "Brier Score": 0.1637, "Accuracy": "76.55%", "Notes": "Linear Baseline"},
                    {"Algorithm": "2. Random Forest (100 Trees)", "ROC-AUC": 0.7070, "Log-Loss": 0.4975, "Brier Score": 0.1624, "Accuracy": "76.74%", "Notes": "Bagging Ensemble"},
                    {"Algorithm": "3. Gradient Boosting + Isotonic Calibration", "ROC-AUC": 0.7076, "Log-Loss": 0.4971, "Brier Score": 0.1623, "Accuracy": "76.76%", "Notes": "Selected Production Model"},
                ]
            )
            st.dataframe(bench_df, width="stretch", hide_index=True)
            st.caption(
                "Evaluated using leak-free GroupShuffleSplit on unseen students, with monotonic_cst=-1 on elapsed time and word difficulty."
            )

    with tab4:
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            with st.container(border=True):
                st.subheader("1. What Problem This Solves")
                st.write(
                    "Traditional flashcard tools rely on rigid, hand-tuned spacing rules (such as repeating every word after 1, 3, and 7 days) regardless of word complexity or past mistakes. "
                    "Reviewing too early wastes study time, while reviewing too late causes memory lapses."
                )
                st.write(
                    "**MemoryLab** replaces fixed rules with a data-driven machine learning scheduler. Given a learner's review timestamps, past recall outcomes, and word difficulty, it forecasts the exact probability of recalling a word across future days and schedules the next review right before retention crosses below your target threshold."
                )

            with st.container(border=True):
                st.subheader("2. Dataset & Feature Engineering")
                st.markdown(
                    """
                    - **Training Corpus:** 1,081,771 cleaned student review logs from the open-source MaiMemo spaced-repetition dataset, merged with lexical difficulty scores (`d`).
                    - **Leak-Free Evaluation:** Split strictly by student ID using `GroupShuffleSplit` (844,860 training rows; 236,911 unseen-user test rows) so the model cannot memorize individual users.
                    - **17 Cognitive Features:** Extracts recall streaks, lapse counts, lapse ratios, recent 3-attempt accuracy, last/mean/maximum intervals, log-transformed elapsed time (`log_delta_t`), and elapsed-to-interval ratios.
                    """
                )

        with col_b:
            with st.container(border=True):
                st.subheader("3. Enforcing Ebbinghaus Forgetting Physics")
                st.write(
                    "In raw observational study logs, students who wait longer between reviews are often advanced learners studying easy words, which can trick unconstrained models into predicting higher recall after longer delays."
                )
                st.write(
                    "To prevent this selection bias, our `HistGradientBoostingClassifier` enforces **monotonic constraints (`monotonic_cst = -1`)** on elapsed time (`delta_t`, `log_delta_t`, `elapsed_to_last_ratio`) and word difficulty (`d`). This mathematically guarantees that predicted retention always decays smoothly as time passes."
                )

            with st.container(border=True):
                st.subheader("4. Probability Calibration & Scheduling Rule")
                st.markdown(
                    """
                    - **Isotonic Calibration:** Raw tree probabilities are calibrated via `CalibratedClassifierCV(method='isotonic')` so a predicted 85% recall score matches true empirical retention rates.
                    - **Optimal Interval Search:** For any word, the scheduler evaluates recall probability $P(r = 1 \\mid \\Delta t)$ for $\\Delta t \\in \\{1, 2, \\dots, T_{\\max}\\}$ and selects the latest day where $P(r = 1 \\mid \\Delta t) \\ge \\tau_{\\text{target}}$.
                    """
                )


if __name__ == "__main__":
    main()
