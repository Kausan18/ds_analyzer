import chromadb
import os
import json
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

chroma_client = chromadb.Client()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def embed_report(session_id: str, report: dict):
    try:
        try:
            chroma_client.delete_collection(f"session_{session_id}")
        except:
            pass

        collection = chroma_client.create_collection(f"session_{session_id}")
        chunks = report_to_chunks(report)

        if chunks["texts"]:
            collection.add(documents=chunks["texts"], ids=chunks["ids"])
    except Exception as e:
        print(f"[ChromaDB] Embed error: {e}")
        raise e


def report_to_chunks(report):
    texts = []
    ids = []

    def add(chunk_id, text):
        safe_id = str(chunk_id).replace(" ", "_").replace("'", "").replace("/", "_")[:60]
        texts.append(str(text))
        ids.append(safe_id)

    # ── Overview ──────────────────────────────────────────────────────────────
    add("overview", (
        f"DATASET OVERVIEW: File '{report['filename']}' has {report['shape']['rows']} rows "
        f"and {report['shape']['cols']} columns. "
        f"{'Analysis was done on a 50,000-row sample from ' + str(report.get('total_rows_original','?')) + ' total rows. ' if report.get('sampled') else ''}"
        f"All columns: {', '.join(report['columns'])}. "
        f"Column data types: {', '.join([f'{col}={dtype}' for col, dtype in report['dtypes'].items()])}."
    ))

    # ── Missing values ────────────────────────────────────────────────────────
    if report["missing"]:
        details = "; ".join([f"'{col}' has {v['count']} missing values ({v['percent']}%)"
                             for col, v in report["missing"].items()])
        add("missing_values", f"MISSING VALUES: {details}. "
            f"Columns with >50% missing should be dropped. Columns with 20-50% missing should be imputed.")
    else:
        add("missing_values", "MISSING VALUES: No missing values found in any column. The dataset is complete.")

    # ── Duplicates ────────────────────────────────────────────────────────────
    dup = report["duplicates"]
    add("duplicates", (
        f"DUPLICATE ROWS: {dup['count']} duplicate rows found, which is {dup['percent']}% of the dataset. "
        f"{'Duplicates should be removed before modelling.' if dup['count'] > 0 else 'No duplicates found — dataset is clean on this front.'}"
    ))

    # ── Outliers ──────────────────────────────────────────────────────────────
    if report["outliers"]:
        details = "; ".join([f"'{col}' has {v['count']} outliers ({v['percent']}% of its values)"
                             for col, v in report["outliers"].items()])
        add("outliers", (
            f"OUTLIERS (detected via Z-score > 3): {details}. "
            f"Outliers can skew model training. Options: cap at 1.5x IQR, remove, or investigate individually. "
            f"High outlier % suggests data quality issues or naturally heavy-tailed distributions."
        ))
    else:
        add("outliers", "OUTLIERS: No significant outliers detected using Z-score threshold of 3.")

    # ── Class imbalance ───────────────────────────────────────────────────────
    if report["class_imbalance"]:
        for col, info in report["class_imbalance"].items():
            dist_str = ", ".join([f"{k}: {v}" for k, v in info["distribution"].items()])
            add(f"imbalance_{col[:40]}",
                f"CLASS IMBALANCE in column '{col}': imbalance ratio is {info['imbalance_ratio']}:1. "
                f"Distribution: {dist_str}. "
                f"{'This column is heavily imbalanced — consider SMOTE oversampling, undersampling, or class_weight parameter in your model.' if info['is_imbalanced'] else 'Imbalance is within acceptable range.'}")
    else:
        add("imbalance", "CLASS IMBALANCE: No significant class imbalance detected in any categorical column.")

    # ── Correlations ──────────────────────────────────────────────────────────
    high_corr = report["correlations"].get("high_correlations", [])
    if high_corr:
        for i, pair in enumerate(high_corr):
            direction = "positively" if pair["correlation"] > 0 else "negatively"
            add(f"correlation_{i}",
                f"HIGH CORRELATION: '{pair['col1']}' and '{pair['col2']}' are strongly {direction} correlated "
                f"(r = {pair['correlation']}). "
                f"This means they carry redundant information. For regression/classification models, "
                f"consider dropping one to reduce multicollinearity. The one with lower correlation "
                f"to the target variable should be dropped.")
    else:
        add("correlations", "CORRELATIONS: No high correlations found between features (threshold |r| > 0.8). Features are relatively independent.")

    # ── Full correlation matrix context ───────────────────────────────────────
    matrix = report["correlations"].get("matrix", {})
    if matrix:
        # Find top 10 most interesting (non-trivial) pairs for context
        pairs = []
        cols = list(matrix.keys())
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = matrix[cols[i]].get(cols[j], 0)
                if val is not None and abs(val) > 0.3:
                    pairs.append((cols[i], cols[j], val))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if pairs:
            summary = "; ".join([f"{a} & {b}: {round(v, 3)}" for a, b, v in pairs[:10]])
            add("correlation_matrix_summary",
                f"CORRELATION SUMMARY (|r| > 0.3): {summary}. "
                f"These moderate-to-strong correlations are worth noting for feature selection.")

    # ── Per-column stats ──────────────────────────────────────────────────────
    numeric_summaries = []
    categorical_summaries = []

    for col, s in report["column_stats"].items():
        if "mean" in s:
            skew_note = ""
            if abs(s["skewness"]) > 1:
                skew_note = f"Highly skewed (skewness={s['skewness']}) — consider log transform. "
            elif abs(s["skewness"]) > 0.5:
                skew_note = f"Moderately skewed (skewness={s['skewness']}). "
            numeric_summaries.append(
                f"'{col}': mean={s['mean']}, median={s['median']}, std={s['std']}, "
                f"min={s['min']}, max={s['max']}, skewness={s['skewness']}, unique={s['unique']}. {skew_note}"
            )
        else:
            top = ", ".join([f"{k}({v})" for k, v in list(s.get("top_values", {}).items())[:5]])
            categorical_summaries.append(
                f"'{col}': {s['unique']} unique values, top values: {top}."
            )

    if numeric_summaries:
        # Split into chunks of 5 columns each to avoid too-long chunks
        for i in range(0, len(numeric_summaries), 5):
            chunk = numeric_summaries[i:i+5]
            add(f"numeric_stats_{i}",
                f"NUMERIC COLUMN STATISTICS: " + " | ".join(chunk))

    if categorical_summaries:
        for i in range(0, len(categorical_summaries), 5):
            chunk = categorical_summaries[i:i+5]
            add(f"categorical_stats_{i}",
                f"CATEGORICAL COLUMN STATISTICS: " + " | ".join(chunk))

    # ── Recommendations ───────────────────────────────────────────────────────
    if report["recommendations"]:
        add("recommendations",
            f"ACTIONABLE RECOMMENDATIONS FROM EDA: " + " | ".join(report["recommendations"]))

    return {"texts": texts, "ids": ids}

# ── Feature importance ────────────────────────────────────────────────────
    fi = report.get("feature_importance", {})
    if fi.get("available"):
        top5 = fi["features"][:5]
        bottom5 = fi["features"][-5:]
        top_str = ", ".join([f"{f['feature']}({f['importance']})" for f in top5])
        bot_str = ", ".join([f"{f['feature']}({f['importance']})" for f in bottom5])
        add("feature_importance",
            f"FEATURE IMPORTANCE (target='{fi['target_column']}', model={fi['model_type']}): "
            f"Top features: {top_str}. "
            f"Least important features: {bot_str}. "
            f"Features with very low importance (<0.01) are candidates for removal.")
    else:
        add("feature_importance",
            f"FEATURE IMPORTANCE: Not available. Reason: {fi.get('reason', 'unknown')}")

    # ── Distributions ─────────────────────────────────────────────────────────
    dist_summary = []
    for col, d in report.get("distributions", {}).items():
        stats_col = report["column_stats"].get(col, {})
        skew = stats_col.get("skewness", 0)
        skew_note = ""
        if abs(skew) > 1:
            skew_note = "highly skewed"
        elif abs(skew) > 0.5:
            skew_note = "moderately skewed"
        else:
            skew_note = "approximately normal"
        dist_summary.append(f"'{col}' distribution is {skew_note} (skewness={skew})")

    if dist_summary:
        add("distributions",
            f"COLUMN DISTRIBUTIONS: " + "; ".join(dist_summary))

def query_report(session_id: str, question: str):
    # Get context from ChromaDB
    try:
        collection = chroma_client.get_collection(f"session_{session_id}")
        results = collection.query(query_texts=[question], n_results=8)
        context_chunks = results["documents"][0]
        context = "\n\n".join(context_chunks)
    except Exception as e:
        print(f"[ChromaDB] Query error: {e}")
        context = "No context available."

    prompt = f"""You are a senior data scientist assistant reviewing an automated EDA (Exploratory Data Analysis) report.
The user has uploaded a dataset and you have access to the full analysis results below.

RULES:
- Answer only based on the EDA context provided
- Be specific: cite exact column names, numbers, and percentages
- Give actionable advice where relevant
- If asked about a chart or visualization, explain what the numbers behind it mean
- If something is not in the context, say "The EDA report doesn't have specific data on that"

EDA REPORT CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Groq] API error: {e}")
        raise Exception(f"Groq API error: {str(e)}")