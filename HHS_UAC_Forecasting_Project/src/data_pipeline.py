import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

CSV_PATH = "user_uploaded_files/ttl=63d/user_uploaded/HHS_Unaccompanied_Alien_Children_Program.csv"

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(how="all").copy()
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "Date": "date",
        "Children apprehended and placed in CBP custody*": "cbp_apprehended",
        "Children apprehended and placed in CBP custody": "cbp_apprehended",
        "Children in CBP custody": "cbp_custody",
        "Children transferred out of CBP custody": "transferred_to_hhs",
        "Children in HHS Care": "hhs_care",
        "Children discharged from HHS Care": "hhs_discharged",
    }
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    num_cols = ["cbp_apprehended", "cbp_custody", "transferred_to_hhs", "hhs_care", "hhs_discharged"]
    for col in num_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").set_index("date")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df["is_observed"] = 1
    df = df.reindex(full_idx)
    df.index.name = "date"
    df["is_observed"] = df["is_observed"].fillna(0).astype(int)
    df[num_cols] = df[num_cols].interpolate(method="time").bfill().ffill()
    return df

def decompose_series(series: pd.Series, period: int = 7) -> pd.DataFrame:
    trend = series.rolling(window=period, center=True, min_periods=1).mean()
    detrended = series - trend
    seasonal = detrended.groupby(series.index.dayofweek).transform("mean")
    residual = series - trend - seasonal
    return pd.DataFrame({"observed": series, "trend": trend, "seasonal": seasonal, "residual": residual}, index=series.index)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["net_flow"] = feat["transferred_to_hhs"] - feat["hhs_discharged"]
    feat["actual_care_change"] = feat["hhs_care"].diff().fillna(0)
    feat["intake_to_exit_ratio"] = (feat["transferred_to_hhs"] + 1.0) / (feat["hhs_discharged"] + 1.0)
    feat["cbp_transfer_efficiency"] = (feat["transferred_to_hhs"] + 1.0) / (feat["cbp_custody"] + 1.0)
    rolling_discharge_7 = feat["hhs_discharged"].rolling(7, min_periods=1).mean().clip(lower=1.0)
    feat["implied_los_days"] = feat["hhs_care"] / rolling_discharge_7

    for col in ["hhs_care", "hhs_discharged", "transferred_to_hhs", "cbp_custody", "cbp_apprehended", "net_flow"]:
        for lag in [1, 2, 3, 7, 14]:
            feat[f"{col}_lag_{lag}"] = feat[col].shift(lag)
        for win in [7, 14]:
            feat[f"{col}_roll_mean_{win}"] = feat[col].shift(1).rolling(win).mean()
            feat[f"{col}_roll_std_{win}"] = feat[col].shift(1).rolling(win).std()

    feat["care_momentum_7d"] = feat["hhs_care"].shift(1) - feat["hhs_care"].shift(8)
    feat["care_momentum_14d"] = feat["hhs_care"].shift(1) - feat["hhs_care"].shift(15)
    feat["dayofweek"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["is_weekend"] = (feat["dayofweek"] >= 5).astype(int)
    feat["sin_dow"] = np.sin(2 * np.pi * feat["dayofweek"] / 7.0)
    feat["cos_dow"] = np.cos(2 * np.pi * feat["dayofweek"] / 7.0)
    return feat.dropna()

if __name__ == "__main__":
    df = load_and_prepare_data(CSV_PATH)
    feat = engineer_features(df)
    print(f"Loaded {len(df)} continuous days ({df['is_observed'].sum()} reported days) from {df.index.min().date()} to {df.index.max().date()}")
    print(f"Engineered {feat.shape[1]} features across {feat.shape[0]} rows.")
    print("\nSummary statistics (raw continuous series):")
    print(df[["cbp_apprehended", "cbp_custody", "transferred_to_hhs", "hhs_care", "hhs_discharged"]].describe().round(1))
    print("\nRecent 5 days:")
    print(df.tail(5)[["cbp_apprehended", "cbp_custody", "transferred_to_hhs", "hhs_care", "hhs_discharged", "is_observed"]])
