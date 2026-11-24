import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TSV_PATH = PROJECT_ROOT / "data/raw/opensource_dataset.tsv"
OUT_PARQUET_PATH = PROJECT_ROOT / "data/processed/maimemo_sample.parquet"

def run_ingestion():
    if not RAW_TSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_TSV_PATH.resolve()}. "
            "Update RAW_TSV_PATH to point to your downloaded opensource_dataset.tsv."
        )

    OUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_csv(RAW_TSV_PATH, separator="\t")
    schema = lf.collect_schema()
    print("=== Detected TSV Schema ===")
    for col, dtype in schema.items():
        print(f"  {col}: {dtype}")


    sample_df = (
        lf.filter(
            pl.col("t_history").is_not_null()
            & pl.col("r_history").is_not_null()
            & (pl.col("t_history") != "")
            & (pl.col("r_history") != "")
            & (pl.col("delta_t") > 0)
        )
        .filter((pl.col("u").hash(seed=42) % 200) == 0)
        .collect()
    )

    sample_df.write_parquet(OUT_PARQUET_PATH)
    print(f"\nSaved {sample_df.height:,} rows across {sample_df['u'].n_unique():,} unique users -> {OUT_PARQUET_PATH}")
    print("\n=== First 5 Sampled Rows ===")
    print(sample_df.head(5))

if __name__ == "__main__":
    run_ingestion()