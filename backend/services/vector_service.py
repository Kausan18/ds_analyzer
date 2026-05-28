import os
import json
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Store full reports in memory for direct injection
report_store = {}


def embed_report(session_id: str, report: dict):
    """Store report for direct context injection — no vector search needed at this scale."""
    report_store[session_id] = report
    print(f"[AI] Report stored for session {session_id}")


def query_report(session_id: str, question: str, history: list = []):
    if session_id not in report_store:
        return "Session not found. Please re-upload your dataset."

    report = report_store[session_id]

    context = build_context_string(report)

    # Build message list: report context + optional history + current question
    messages = [
        {"role": "user", "content": f"EDA REPORT CONTEXT:\n{context}\n\nUse this report to answer my questions."},
        {"role": "assistant", "content": "Understood. I have reviewed the EDA report. Ask me anything about your dataset."},
    ]
    for msg in history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": question})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=800,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Groq] API error: {e}")
        raise Exception(f"Groq API error: {str(e)}")


def build_context_string(report: dict) -> str:
    lines = []

    lines.append(f"FILE: {report['filename']} | ROWS: {report['shape']['rows']} | COLS: {report['shape']['cols']}")
    lines.append(f"COLUMNS: {', '.join(report['columns'])}")

    if report["missing"]:
        miss = ", ".join([f"{c}({v['percent']}%)" for c, v in report["missing"].items()])
        lines.append(f"MISSING VALUES: {miss}")
    else:
        lines.append("MISSING VALUES: None")

    dup = report["duplicates"]
    lines.append(f"DUPLICATES: {dup['count']} rows ({dup['percent']}%)")

    if report["outliers"]:
        out = ", ".join([f"{c}({v['count']} outliers, {v['percent']}%)" for c, v in report["outliers"].items()])
        lines.append(f"OUTLIERS: {out}")
    else:
        lines.append("OUTLIERS: None detected")

    if report["class_imbalance"]:
        for col, info in report["class_imbalance"].items():
            lines.append(f"CLASS IMBALANCE '{col}': ratio {info['imbalance_ratio']}:1, imbalanced={info['is_imbalanced']}, dist={info['distribution']}")

    high_corr = report["correlations"].get("high_correlations", [])
    if high_corr:
        pairs = ", ".join([f"{p['col1']}&{p['col2']}(r={p['correlation']})" for p in high_corr])
        lines.append(f"HIGH CORRELATIONS: {pairs}")
    else:
        lines.append("HIGH CORRELATIONS: None above 0.8")

    for col, s in report["column_stats"].items():
        if "mean" in s:
            lines.append(f"STAT '{col}': mean={s['mean']}, std={s['std']}, min={s['min']}, max={s['max']}, skew={s['skewness']}, unique={s['unique']}, q25={s.get('q25','')}, q75={s.get('q75','')}")
        else:
            top = list(s.get("top_values", {}).items())[:3]
            lines.append(f"STAT '{col}' (categorical): unique={s['unique']}, top={top}")

    fi = report.get("feature_importance", {})
    if fi.get("available"):
        top5 = fi["features"][:5]
        bot3 = fi["features"][-3:]
        lines.append(f"FEATURE IMPORTANCE (target='{fi['target_column']}', model={fi['model_type']}): top={top5}, lowest={bot3}")

    dist_notes = []
    for col, d in report.get("distributions", {}).items():
        s = report["column_stats"].get(col, {})
        skew = s.get("skewness", 0)
        if abs(skew) > 0.5:
            dist_notes.append(f"{col}(skew={skew})")
    if dist_notes:
        lines.append(f"SKEWED COLUMNS: {', '.join(dist_notes)}")

    lines.append(f"RECOMMENDATIONS: {' | '.join(report.get('recommendations', []))}")

    return "\n".join(lines)