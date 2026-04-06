import pandas as pd
import numpy as np
import uuid
import io
from scipy import stats

# ── Numpy type converter ─────────────────────────────────────────────────────
def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def run_eda(file_bytes: bytes, filename: str):
    session_id = str(uuid.uuid4())[:8]

    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    sampled = False
    original_rows = int(df.shape[0])
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        sampled = True

    report = {
        "session_id": session_id,
        "filename": filename,
        "sampled": sampled,
        "total_rows_original": original_rows,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": get_missing(df),
        "duplicates": get_duplicates(df),
        "outliers": get_outliers(df),
        "class_imbalance": get_class_imbalance(df),
        "correlations": get_correlations(df),
        "column_stats": get_column_stats(df),
        "recommendations": []
    }

    report["recommendations"] = generate_recommendations(report)

    # Convert ALL numpy types before returning
    return session_id, convert(report)


def get_missing(df):
    missing = {}
    for col in df.columns:
        count = int(df[col].isnull().sum())
        if count > 0:
            missing[col] = {
                "count": count,
                "percent": round(float(count / len(df) * 100), 2)
            }
    return missing


def get_duplicates(df):
    dup_count = int(df.duplicated().sum())
    return {
        "count": dup_count,
        "percent": round(float(dup_count / len(df) * 100), 2)
    }


def get_outliers(df):
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue
        z_scores = np.abs(stats.zscore(col_data))
        outlier_count = int((z_scores > 3).sum())
        if outlier_count > 0:
            outliers[col] = {
                "count": outlier_count,
                "percent": round(float(outlier_count / len(col_data) * 100), 2)
            }
    return outliers


def get_class_imbalance(df):
    imbalance = {}
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        counts = df[col].value_counts()
        if len(counts) < 2 or len(counts) > 20:
            continue
        ratio = round(float(counts.iloc[0] / counts.iloc[-1]), 2)
        imbalance[col] = {
            "distribution": {str(k): int(v) for k, v in counts.items()},
            "imbalance_ratio": ratio,
            "is_imbalanced": bool(ratio > 5)
        }
    return imbalance


def get_correlations(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {}
    corr = numeric_df.corr()
    high_corr = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            if abs(float(val)) > 0.8:
                high_corr.append({
                    "col1": str(cols[i]),
                    "col2": str(cols[j]),
                    "correlation": round(float(val), 3)
                })
    return {
        "matrix": {
            str(col): {str(c): round(float(v), 3) if not pd.isna(v) else 0.0
                       for c, v in row.items()}
            for col, row in corr.to_dict().items()
        },
        "high_correlations": high_corr
    }


def get_column_stats(df):
    stats_dict = {}
    for col in df.columns:
        col_data = df[col].dropna()
        info = {
            "dtype": str(df[col].dtype),
            "unique": int(df[col].nunique())
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            info.update({
                "mean": round(float(col_data.mean()), 4) if len(col_data) > 0 else 0.0,
                "median": round(float(col_data.median()), 4) if len(col_data) > 0 else 0.0,
                "std": round(float(col_data.std()), 4) if len(col_data) > 0 else 0.0,
                "min": round(float(col_data.min()), 4) if len(col_data) > 0 else 0.0,
                "max": round(float(col_data.max()), 4) if len(col_data) > 0 else 0.0,
                "skewness": round(float(col_data.skew()), 4) if len(col_data) > 0 else 0.0
            })
        else:
            info["top_values"] = {
                str(k): int(v)
                for k, v in col_data.value_counts().head(5).items()
            }
        stats_dict[col] = info
    return stats_dict


def generate_recommendations(report):
    recs = []

    for col, info in report["missing"].items():
        if info["percent"] > 50:
            recs.append(f"DROP column '{col}' — {info['percent']}% missing values, too sparse to impute.")
        elif info["percent"] > 20:
            recs.append(f"IMPUTE column '{col}' with median/mode — {info['percent']}% missing.")
        else:
            recs.append(f"Consider filling '{col}' ({info['percent']}% missing) with mean or forward-fill.")

    if report["duplicates"]["percent"] > 5:
        recs.append(f"Remove {report['duplicates']['count']} duplicate rows ({report['duplicates']['percent']}% of dataset).")

    for col, info in report["outliers"].items():
        recs.append(f"Column '{col}' has {info['count']} outliers ({info['percent']}%) — consider capping or investigation.")

    for col, info in report["class_imbalance"].items():
        if info["is_imbalanced"]:
            recs.append(f"Column '{col}' is heavily imbalanced (ratio {info['imbalance_ratio']}:1) — consider SMOTE or class weighting.")

    for pair in report["correlations"].get("high_correlations", []):
        recs.append(f"HIGH CORRELATION ({pair['correlation']}) between '{pair['col1']}' and '{pair['col2']}' — consider dropping one to reduce multicollinearity.")

    if not recs:
        recs.append("Dataset looks clean. No major issues detected.")

    return recs