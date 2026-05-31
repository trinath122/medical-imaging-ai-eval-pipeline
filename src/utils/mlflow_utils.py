"""
MLflow helper utilities.
"""
import mlflow
import pandas as pd
from pathlib import Path
from typing import Dict


def log_metrics_dict(metrics: Dict[str, float], step: int = 0) -> None:
    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)


def log_dataframe_artifact(df: pd.DataFrame, filename: str, results_dir: str = "results") -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(results_dir) / filename)
    df.to_csv(path)
    mlflow.log_artifact(path)


def log_failure_report(df: pd.DataFrame, results_dir: str = "results") -> None:
    log_dataframe_artifact(df, "failure_modes.csv", results_dir)
    # Log summary counts as metrics
    if not df.empty:
        for mode in ["FALSE_POSITIVE_HIGH_CONF", "FALSE_NEGATIVE_HIGH_CONF",
                     "FALSE_POSITIVE_UNCERTAIN", "FALSE_NEGATIVE_UNCERTAIN",
                     "RARE_CLASS_MISS", "MULTI_LABEL_CONFUSION"]:
            count = df["failure_modes"].str.contains(mode, na=False).sum()
            mlflow.log_metric(f"failure/{mode}", int(count))
