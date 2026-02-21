"""
data_utils.py — Data loading, preprocessing pipelines, and named feature sets.

All heavy data loading should go through load_data() for consistency.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "playground-series-s6e2"

# ── Schema ─────────────────────────────────────────────────────────────────────
TARGET = "Heart Disease"
TARGET_MAP = {"Presence": 1, "Absence": 0}
TARGET_MAP_INV = {v: k for k, v in TARGET_MAP.items()}

FEATURE_COLS = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
    "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium",
]

# Feature type classification
CONTINUOUS_COLS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
BINARY_COLS = ["Sex", "FBS over 120", "Exercise angina"]
ORDINAL_COLS = ["Chest pain type", "EKG results", "Slope of ST",
                "Number of vessels fluro", "Thallium"]


def load_data(
    split: str = "train",
    encode_target: bool = True,
    dtype_optimize: bool = True,
) -> pd.DataFrame:
    """Load train or test CSV and optionally encode the target to 0/1.

    Parameters
    ----------
    split : 'train' | 'test'
    encode_target : if True, map Presence→1, Absence→0
    dtype_optimize : downcast numerics to float32/int8 to save RAM

    Returns
    -------
    pd.DataFrame
    """
    path = DATA_DIR / f"{split}.csv"
    df = pd.read_csv(path)

    if encode_target and TARGET in df.columns:
        df[TARGET] = df[TARGET].map(TARGET_MAP).astype(np.int8)

    if dtype_optimize:
        for col in CONTINUOUS_COLS:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        for col in BINARY_COLS + ORDINAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(np.int8)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-inspired interaction and derived features (in-place copy)."""
    df = df.copy()
    df["Age_HR_ratio"] = df["Age"] / (df["Max HR"] + 1e-6)
    df["BP_Chol_product"] = df["BP"] * df["Cholesterol"]
    df["ST_Slope_interaction"] = df["ST depression"] * df["Slope of ST"]
    df["High_risk_age"] = (df["Age"] > 55).astype(np.int8)
    # Absolute ST depression magnitude (some rows may have been negative in source)
    df["ST_abs"] = df["ST depression"].abs()
    return df


def get_X_y(
    df: pd.DataFrame,
    extra_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from a dataframe."""
    cols = FEATURE_COLS.copy()
    if extra_features:
        df = add_engineered_features(df)
        cols += ["Age_HR_ratio", "BP_Chol_product", "ST_Slope_interaction",
                 "High_risk_age", "ST_abs"]
    X = df[cols]
    y = df[TARGET]
    return X, y


def get_feature_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return a dict of named feature sets for ablation studies.

    Keys
    ----
    'baseline'    : raw features, no scaling
    'scaled'      : one-hot encoded + StandardScaler
    'engineered'  : + interaction features
    'pca'         : PCA-reduced (fit inline, educational — redo properly in notebook)
    """
    from sklearn.decomposition import PCA

    X_base, _ = get_X_y(df, extra_features=False)
    X_eng, _ = get_X_y(df, extra_features=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_base),
        columns=X_base.columns,
        index=X_base.index,
    )
    X_eng_scaled = pd.DataFrame(
        scaler.fit_transform(X_eng),
        columns=X_eng.columns,
        index=X_eng.index,
    )

    pca = PCA(n_components=0.95, random_state=42)
    X_pca_arr = pca.fit_transform(X_scaled)
    X_pca = pd.DataFrame(
        X_pca_arr,
        columns=[f"PC{i+1}" for i in range(X_pca_arr.shape[1])],
        index=X_base.index,
    )

    return {
        "baseline": X_base,
        "scaled": X_scaled,
        "engineered": X_eng_scaled,
        "pca": X_pca,
    }
