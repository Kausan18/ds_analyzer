import pandas as pd
import numpy as np
import uuid
import io
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
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
        "distributions": get_distributions(df),
        "feature_importance": get_feature_importance(df),
        "dataset_preview": get_preview(df),
        "recommendations": []
    }

    report["recommendations"] = generate_recommendations(report)
    return session_id, convert(report)


def get_preview(df):
    """First 10 rows as a JSON-serializable list."""
    preview_df = df.head(10).copy()
    # Convert any non-serializable types
    for col in preview_df.columns:
        preview_df[col] = preview_df[col].astype(str)
    return {
        "columns": list(preview_df.columns),
        "rows": preview_df.values.tolist()
    }


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
                "skewness": round(float(col_data.skew()), 4) if len(col_data) > 0 else 0.0,
                "q25": round(float(col_data.quantile(0.25)), 4) if len(col_data) > 0 else 0.0,
                "q75": round(float(col_data.quantile(0.75)), 4) if len(col_data) > 0 else 0.0,
            })
        else:
            info["top_values"] = {
                str(k): int(v)
                for k, v in col_data.value_counts().head(5).items()
            }
        stats_dict[col] = info
    return stats_dict


def get_distributions(df):
    """
    For each numeric column, compute histogram bin data (counts + edges).
    Returns data only — charts are rendered in frontend/PDF from this.
    """
    distributions = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 5:
            continue
        counts, bin_edges = np.histogram(col_data, bins=30)
        distributions[col] = {
            "counts": counts.tolist(),
            "bin_edges": [round(float(e), 4) for e in bin_edges.tolist()],
            "bin_centers": [round(float((bin_edges[i] + bin_edges[i+1]) / 2), 4)
                           for i in range(len(bin_edges) - 1)]
        }
    return distributions


def get_feature_importance(df):
    """
    Auto-detect the most likely target column and compute
    feature importance using a RandomForest.
    """
    try:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.shape[1] < 2:
            return {"available": False, "reason": "Not enough numeric columns"}

        # Drop columns with >30% missing
        numeric_df = numeric_df.dropna(thresh=int(len(numeric_df) * 0.7), axis=1)
        numeric_df = numeric_df.fillna(numeric_df.median())

        # Auto-detect target: prefer columns named target/label/class/y/output/result
        target_keywords = ["target", "label", "class", "output", "result",
                          "y", "churn", "salary", "price", "score", "fraud"]
        target_col = None
        for col in numeric_df.columns:
            if any(kw in col.lower() for kw in target_keywords):
                target_col = col
                break

        # If no keyword match, use the last numeric column
        if target_col is None:
            target_col = numeric_df.columns[-1]

        feature_cols = [c for c in numeric_df.columns if c != target_col]
        if len(feature_cols) < 1:
            return {"available": False, "reason": "Not enough feature columns"}

        X = numeric_df[feature_cols]
        y = numeric_df[target_col]

        # Use classifier if few unique values, regressor otherwise
        n_unique = y.nunique()
        if n_unique <= 10:
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        model.fit(X, y)
        importances = model.feature_importances_

        importance_list = sorted(
            [{"feature": col, "importance": round(float(imp), 4)}
             for col, imp in zip(feature_cols, importances)],
            key=lambda x: x["importance"],
            reverse=True
        )

        return {
            "available": True,
            "target_column": target_col,
            "model_type": "classifier" if n_unique <= 10 else "regressor",
            "features": importance_list
        }

    except Exception as e:
        return {"available": False, "reason": str(e)}


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
        recs.append(f"HIGH CORRELATION ({pair['correlation']}) between '{pair['col1']}' and '{pair['col2']}' — consider dropping one.")

    # Skewness recommendations
    for col, s in report["column_stats"].items():
        if "skewness" in s and abs(s["skewness"]) > 1:
            recs.append(f"Column '{col}' is highly skewed (skewness={s['skewness']}) — consider log or Box-Cox transform.")

    # Feature importance recommendation
    fi = report.get("feature_importance", {})
    if fi.get("available") and fi.get("features"):
        top = fi["features"][0]
        bottom = fi["features"][-1]
        recs.append(f"Most important feature for '{fi['target_column']}': '{top['feature']}' (importance={top['importance']}). Least important: '{bottom['feature']}' (importance={bottom['importance']}) — consider dropping.")

    if not recs:
        recs.append("Dataset looks clean. No major issues detected.")

    return recs