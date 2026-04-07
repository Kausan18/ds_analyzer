import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

BACKEND = "http://localhost:8000/api"

st.set_page_config(page_title="Dataset Quality Evaluator", layout="wide")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-label { font-size: 12px; color: #aaa; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 700; }
.good { color: #4ade80; }
.warn { color: #fbbf24; }
.bad  { color: #f87171; }
</style>
""", unsafe_allow_html=True)

st.title("Dataset Quality Evaluator")
st.caption("Upload a CSV or Excel file to get a full automated EDA report with AI-powered insights.")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file and "report" not in st.session_state:
    progress = st.progress(0, text="Reading file...")
    try:
        progress.progress(20, text="Sending to analysis engine...")
        response = requests.post(
            f"{BACKEND}/analyze",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=300
        )
        progress.progress(80, text="Processing results...")

        if response.status_code == 200:
            data = response.json()
            st.session_state["report"] = data["report"]
            st.session_state["session_id"] = data["session_id"]
            progress.progress(100, text="Done!")
            if data["report"].get("sampled"):
                st.warning(
                    f"Large dataset detected — {data['report']['total_rows_original']:,} rows. "
                    f"Analysis ran on a 50,000-row sample."
                )
            st.success("Analysis complete!")
        else:
            progress.empty()
            try:
                detail = response.json().get("detail", "Unknown error")
            except Exception:
                detail = f"HTTP {response.status_code}"
            st.error(f"Error: {detail}")

    except requests.exceptions.Timeout:
        progress.empty()
        st.error("Request timed out. Try a smaller file.")
    except requests.exceptions.ConnectionError:
        progress.empty()
        st.error("Cannot reach backend. Is uvicorn running on port 8000?")

if "report" not in st.session_state:
    st.stop()

report = st.session_state["report"]
session_id = st.session_state["session_id"]

# ── Summary metric cards ──────────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4, c5, c6 = st.columns(6)

def color_class(value, warn_thresh, bad_thresh):
    if value == 0:
        return "good"
    elif value < warn_thresh:
        return "warn"
    return "bad"

with c1:
    st.metric("Rows", f"{report['shape']['rows']:,}")
with c2:
    st.metric("Columns", report["shape"]["cols"])
with c3:
    miss_count = len(report["missing"])
    st.metric("Missing Cols", miss_count, delta=f"{'Clean' if miss_count == 0 else 'Issues found'}", delta_color="normal" if miss_count == 0 else "inverse")
with c4:
    dup = report["duplicates"]["count"]
    st.metric("Duplicates", dup, delta=f"{'Clean' if dup == 0 else str(report['duplicates']['percent'])+'%'}", delta_color="normal" if dup == 0 else "inverse")
with c5:
    out_count = len(report["outliers"])
    st.metric("Outlier Cols", out_count)
with c6:
    fi = report.get("feature_importance", {})
    if fi.get("available"):
        st.metric("Target Col", fi["target_column"])
    else:
        st.metric("Target Col", "N/A")

st.divider()

# ── Tab navigation (persisted) ────────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0

tab_names = [
    "Preview", "Missing Values", "Outliers",
    "Distributions", "Class Imbalance", "Correlations",
    "Feature Importance", "Column Stats", "Recommendations", "AI Assistant"
]

selected_tab = st.radio(
    label="",
    options=tab_names,
    index=st.session_state["active_tab"],
    horizontal=True,
    key="tab_selector",
    label_visibility="collapsed"
)
st.session_state["active_tab"] = tab_names.index(selected_tab)
st.divider()

# ── Preview ───────────────────────────────────────────────────────────────────
if selected_tab == "Preview":
    st.subheader("Dataset Preview")
    preview = report.get("dataset_preview", {})
    if preview:
        preview_df = pd.DataFrame(preview["rows"], columns=preview["columns"])
        st.dataframe(preview_df, use_container_width=True)
        st.caption(f"Showing first 10 rows of {report['shape']['rows']:,} total rows across {report['shape']['cols']} columns.")
    else:
        st.info("Preview not available.")

    st.subheader("Column Data Types")
    dtype_df = pd.DataFrame([
        {"Column": col, "Type": dtype,
         "Category": "Numeric" if "int" in dtype or "float" in dtype else "Categorical" if dtype == "object" else "Other"}
        for col, dtype in report["dtypes"].items()
    ])
    fig = px.pie(dtype_df, names="Category", title="Column Type Distribution", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dtype_df, use_container_width=True)

# ── Missing Values ────────────────────────────────────────────────────────────
elif selected_tab == "Missing Values":
    st.subheader("Missing Values")
    if report["missing"]:
        miss_df = pd.DataFrame([
            {"Column": k, "Missing Count": v["count"], "Missing %": v["percent"]}
            for k, v in report["missing"].items()
        ]).sort_values("Missing %", ascending=False)

        fig = px.bar(
            miss_df, x="Column", y="Missing %",
            color="Missing %", color_continuous_scale="Reds",
            title="Missing Value % per Column",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(miss_df, use_container_width=True)
    else:
        st.success("No missing values found. Dataset is complete.")

# ── Outliers ──────────────────────────────────────────────────────────────────
elif selected_tab == "Outliers":
    st.subheader("Outliers (Z-score > 3)")
    if report["outliers"]:
        out_df = pd.DataFrame([
            {"Column": k, "Outlier Count": v["count"], "Outlier %": v["percent"]}
            for k, v in report["outliers"].items()
        ]).sort_values("Outlier %", ascending=False)

        fig = px.bar(
            out_df, x="Column", y="Outlier %",
            color="Outlier %", color_continuous_scale="Oranges",
            title="Outlier % per Column",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out_df, use_container_width=True)
    else:
        st.success("No significant outliers detected.")

# ── Distributions ─────────────────────────────────────────────────────────────
elif selected_tab == "Distributions":
    st.subheader("Column Distributions")
    distributions = report.get("distributions", {})
    if distributions:
        cols_list = list(distributions.keys())
        selected_cols = st.multiselect(
            "Select columns to view (default: all)",
            options=cols_list,
            default=cols_list[:6]
        )
        if not selected_cols:
            selected_cols = cols_list[:6]

        n = len(selected_cols)
        n_cols = 2
        n_rows = (n + 1) // n_cols

        for row_i in range(n_rows):
            cols = st.columns(n_cols)
            for col_i in range(n_cols):
                idx = row_i * n_cols + col_i
                if idx >= n:
                    break
                col_name = selected_cols[idx]
                d = distributions[col_name]
                s = report["column_stats"].get(col_name, {})

                with cols[col_i]:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=d["bin_centers"],
                        y=d["counts"],
                        name=col_name,
                        marker_color="steelblue"
                    ))
                    # Add mean line
                    if "mean" in s:
                        fig.add_vline(
                            x=s["mean"], line_dash="dash",
                            line_color="red", annotation_text=f"Mean: {s['mean']}"
                        )
                    fig.update_layout(
                        title=f"{col_name}",
                        template="plotly_white",
                        height=280,
                        margin=dict(t=40, b=30, l=30, r=10),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if "skewness" in s:
                        skew = s["skewness"]
                        if abs(skew) > 1:
                            st.warning(f"Highly skewed: {skew}")
                        elif abs(skew) > 0.5:
                            st.info(f"Moderately skewed: {skew}")
                        else:
                            st.success(f"Normal-ish: skew={skew}")
    else:
        st.info("No numeric columns found for distribution analysis.")

# ── Class Imbalance ───────────────────────────────────────────────────────────
elif selected_tab == "Class Imbalance":
    st.subheader("Class Imbalance")
    if report["class_imbalance"]:
        for col, info in report["class_imbalance"].items():
            badge = "🔴 Imbalanced" if info["is_imbalanced"] else "🟢 Acceptable"
            st.markdown(f"**{col}** — Ratio: `{info['imbalance_ratio']}:1`  {badge}")
            dist_df = pd.DataFrame(
                list(info["distribution"].items()),
                columns=["Class", "Count"]
            )
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(dist_df, names="Class", values="Count",
                             title=f"Distribution of {col}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(dist_df, x="Class", y="Count",
                             title=f"Class Counts — {col}", template="plotly_white",
                             color="Count", color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)
            st.divider()
    else:
        st.success("No class imbalance issues found.")

# ── Correlations ──────────────────────────────────────────────────────────────
elif selected_tab == "Correlations":
    st.subheader("Correlation Matrix")
    matrix = report["correlations"].get("matrix", {})
    if matrix:
        corr_df = pd.DataFrame(matrix)
        fig = px.imshow(
            corr_df, text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
            zmin=-1, zmax=1,
            template="plotly_white"
        )
        fig.update_layout(height=max(400, len(corr_df.columns) * 40))
        st.plotly_chart(fig, use_container_width=True)

        high = report["correlations"].get("high_correlations", [])
        if high:
            st.warning(f"**{len(high)} highly correlated feature pairs detected (|r| > 0.8):**")
            high_df = pd.DataFrame(high)
            high_df["abs_corr"] = high_df["correlation"].abs()
            high_df = high_df.sort_values("abs_corr", ascending=False).drop("abs_corr", axis=1)
            st.dataframe(high_df, use_container_width=True)
        else:
            st.success("No high correlations found.")
    else:
        st.info("Not enough numeric columns for correlation analysis.")

# ── Feature Importance ────────────────────────────────────────────────────────
elif selected_tab == "Feature Importance":
    st.subheader("Feature Importance")
    fi = report.get("feature_importance", {})

    if fi.get("available"):
        st.info(f"Target column: **{fi['target_column']}** | Model: **Random Forest {fi['model_type'].title()}**")

        fi_df = pd.DataFrame(fi["features"])
        fi_df["color"] = fi_df["importance"].apply(
            lambda x: "High" if x > 0.1 else ("Medium" if x > 0.03 else "Low")
        )

        fig = px.bar(
            fi_df, x="importance", y="feature",
            orientation="h",
            color="color",
            color_discrete_map={"High": "#4ade80", "Medium": "#fbbf24", "Low": "#f87171"},
            title=f"Feature Importance for predicting '{fi['target_column']}'",
            template="plotly_white",
            labels={"importance": "Importance Score", "feature": "Feature"}
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(400, len(fi_df) * 30),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Importance Table")
        fi_df["recommendation"] = fi_df["importance"].apply(
            lambda x: "Keep — high signal" if x > 0.1
            else ("Keep — moderate signal" if x > 0.03
            else "Consider dropping — low signal")
        )
        st.dataframe(fi_df[["feature", "importance", "recommendation"]], use_container_width=True)
    else:
        st.warning(f"Feature importance not available: {fi.get('reason', 'Unknown error')}")

# ── Column Stats ──────────────────────────────────────────────────────────────
elif selected_tab == "Column Stats":
    st.subheader("Column Statistics")

    numeric_rows = []
    cat_rows = []
    for col, s in report["column_stats"].items():
        if "mean" in s:
            numeric_rows.append({
                "Column": col, "Mean": s["mean"], "Median": s["median"],
                "Std": s["std"], "Min": s["min"], "Max": s["max"],
                "Skewness": s["skewness"], "Q25": s.get("q25", ""),
                "Q75": s.get("q75", ""), "Unique": s["unique"]
            })
        else:
            top = ", ".join([f"{k}({v})" for k, v in list(s.get("top_values", {}).items())[:3]])
            cat_rows.append({"Column": col, "Unique Values": s["unique"], "Top Values": top})

    if numeric_rows:
        st.markdown("**Numeric Columns**")
        st.dataframe(pd.DataFrame(numeric_rows), use_container_width=True)

        # Skewness chart
        skew_df = pd.DataFrame(numeric_rows)[["Column", "Skewness"]]
        fig = px.bar(
            skew_df, x="Column", y="Skewness",
            color="Skewness", color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            title="Skewness per Column",
            template="plotly_white"
        )
        fig.add_hline(y=1, line_dash="dash", line_color="orange", annotation_text="+1 threshold")
        fig.add_hline(y=-1, line_dash="dash", line_color="orange", annotation_text="-1 threshold")
        st.plotly_chart(fig, use_container_width=True)

    if cat_rows:
        st.markdown("**Categorical Columns**")
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

# ── Recommendations ───────────────────────────────────────────────────────────
elif selected_tab == "Recommendations":
    st.subheader("Actionable Recommendations")
    if report["recommendations"]:
        for i, rec in enumerate(report["recommendations"]):
            if rec.startswith("DROP"):
                st.error(f"🗑️ **{i+1}.** {rec}")
            elif any(rec.startswith(w) for w in ["IMPUTE", "Consider", "HIGH CORRELATION"]):
                st.warning(f"⚠️ **{i+1}.** {rec}")
            elif "skewed" in rec.lower():
                st.info(f"📊 **{i+1}.** {rec}")
            elif "important" in rec.lower():
                st.success(f"🎯 **{i+1}.** {rec}")
            else:
                st.info(f"💡 **{i+1}.** {rec}")
    else:
        st.success("Dataset looks clean! No major issues found.")

# ── AI Assistant ──────────────────────────────────────────────────────────────
elif selected_tab == "AI Assistant":
    st.subheader("Ask the AI about your dataset")
    st.caption("The AI has read your full EDA report including distributions, feature importance, correlations, and all statistics.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested questions
    if not st.session_state["chat_history"]:
        st.markdown("**Suggested questions:**")
        suggestions = [
            "Which features should I drop before modelling?",
            "Explain the correlation heatmap",
            "Which columns need log transformation?",
            "What is the most important feature?",
            "Summarize all the issues found in this dataset"
        ]
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["chat_history"].append({"role": "user", "content": suggestion})
                    with st.spinner("Thinking..."):
                        try:
                            res = requests.post(
                                f"{BACKEND}/chat",
                                json={"session_id": session_id, "question": suggestion}
                            )
                            answer = res.json()["answer"] if res.status_code == 200 else f"Error {res.status_code}"
                        except Exception as e:
                            answer = f"Could not reach backend: {str(e)}"
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                    st.rerun()
        st.divider()

    # Chat history display
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    question = st.chat_input("Ask anything about your dataset analysis...")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{BACKEND}/chat",
                    json={"session_id": session_id, "question": question}
                )
                answer = res.json()["answer"] if res.status_code == 200 else f"Error {res.status_code}"
            except Exception as e:
                answer = f"Could not reach backend: {str(e)}"
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()

    # Clear chat button
    if st.session_state["chat_history"]:
        if st.button("Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()

# ── Download PDF ──────────────────────────────────────────────────────────────
st.divider()
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF with all charts..."):
            pdf_res = requests.get(f"{BACKEND}/download/{session_id}")
        if pdf_res.status_code == 200:
            st.download_button(
                label="Download PDF",
                data=pdf_res.content,
                file_name=f"eda_report_{session_id}.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Failed to generate PDF. Check backend terminal.")
with col2:
    if st.button("Start New Analysis"):
        for key in ["report", "session_id", "chat_history", "active_tab"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()