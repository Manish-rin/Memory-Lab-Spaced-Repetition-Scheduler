
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PARQUET = PROJECT_ROOT / "data/processed/maimemo_sample.parquet"
DIFF_TSV = PROJECT_ROOT / "data/raw/opensource_dataset_difficulty.tsv"
FEAT_PARQUET = PROJECT_ROOT / "data/processed/features_1m.parquet"

def build_features():
    df = pl.read_parquet(SAMPLE_PARQUET)


    if DIFF_TSV.exists():
        diff_df = pl.read_csv(DIFF_TSV, separator="\t")
        print("=== Difficulty TSV Schema & Sample ===")
        print(diff_df.head(5))
        
        diff_cols = [c for c in diff_df.columns if c != "w"]
        diff_col = diff_cols[0]
        diff_clean = diff_df.select(["w", pl.col(diff_col).cast(pl.Float64).alias("d")])
        
        median_d = diff_clean["d"].median()
        df = df.join(diff_clean, on="w", how="left").with_columns(
            pl.col("d").fill_null(median_d)
        )
    else:
        print("Warning: opensource_dataset_difficulty.tsv not found in data/raw/. Computing empirical word fail rate as difficulty proxy.")
        word_stats = (
            df.group_by("w")
            .agg((1.0 - pl.col("r").mean()).alias("d"))
        )
        df = df.join(word_stats, on="w", how="left")

    df = df.with_columns([
        pl.col("t_history").str.split(",").list.eval(pl.element().cast(pl.Float64)).alias("t_list"),
        pl.col("r_history").str.split(",").list.eval(pl.element().cast(pl.Int64)).alias("r_list"),
    ])

    features = df.with_columns([
        # History counts & lapse ratios
        pl.col("r_list").list.len().alias("hist_len"),
        pl.col("r_list").list.sum().alias("recall_count"),
        (pl.col("r_list").list.len() - pl.col("r_list").list.sum()).alias("lapse_count"),
        (1.0 - pl.col("r_list").list.mean()).alias("lapse_ratio"),
        pl.col("r_list").list.last().alias("last_outcome"),


        pl.col("r_list").list.tail(3).list.mean().alias("recent_3_recall_rate"),


        pl.col("t_list").list.last().alias("last_interval"),
        pl.col("t_list").list.mean().alias("mean_interval"),
        pl.col("t_list").list.max().alias("max_interval"),

        (pl.col("delta_t") + 1.0).log().alias("log_delta_t"),
        (pl.col("t_list").list.last() + 1.0).log().alias("log_last_interval"),
        (pl.col("delta_t") / (pl.col("t_list").list.last() + 0.1)).alias("elapsed_to_last_ratio"),
        (pl.col("delta_t") / (pl.col("t_list").list.max() + 0.1)).alias("elapsed_to_max_ratio"),

        pl.col("w").str.len_chars().alias("word_len"),
    ]).drop(["t_list", "r_list", "t_history", "r_history"])

    features.write_parquet(FEAT_PARQUET)
    print(f"\nSaved {features.height:,} rows x {len(features.columns)} columns -> {FEAT_PARQUET}")
    print("\n=== Engineered Feature Sample ===")
    print(features.head(5))

if __name__ == "__main__":
    build_features()