import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

BACKEND = "http://localhost:8000/api"

st.set_page_config(page_title="Dataset Evaluator", layout="wide")
st.title("Dataset Quality Evaluator")

# ── Upload ──────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file and "report" not in st.session_state:
    with st.spinner("Running analysis... (large files may take 30–60 seconds)"):
        try:
            response = requests.post(
                f"{BACKEND}/analyze",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=180  # 3 minute timeout for large files
            )
            
            # Check if response has content before parsing JSON
            if response.status_code == 200:
                try:
                    data = response.json()
                    st.session_state["report"] = data["report"]
                    st.session_state["session_id"] = data["session_id"]
                    if data["report"].get("sampled"):
                        st.warning(f"Dataset has {data['report']['total_rows_original']:,} rows — analysis ran on a 50,000-row sample for performance.")
                    st.success("Analysis complete!")
                except requests.exceptions.JSONDecodeError:
                    st.error("Backend returned an invalid response. Check your terminal for the error.")
            else:
                # Try to get error detail, fall back to status code
                try:
                    detail = response.json().get("detail", "Unknown error")
                except Exception:
                    detail = f"HTTP {response.status_code} — check backend terminal for traceback"
                st.error(f"Error: {detail}")

        except requests.exceptions.Timeout:
            st.error("Request timed out. Your file may be too large — try a smaller sample.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend. Is `uvicorn main:app --reload --port 8000` running?")

if "report" not in st.session_state:
    st.stop()

report = st.session_state["report"]
session_id = st.session_state["session_id"]

# ── Overview cards ───────────────────────────────────────────────────────────
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", report["shape"]["rows"])
c2.metric("Columns", report["shape"]["cols"])
c3.metric("Missing Cols", len(report["missing"]))
c4.metric("Duplicate Rows", report["duplicates"]["count"])

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
# ── Tab state persistence ─────────────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0

tab_names = ["Missing Values", "Outliers", "Class Imbalance", "Correlations", "Column Stats", "Recommendations", "AI Assistant"]

# Radio used as tab selector (hidden styling via markdown)
st.markdown("""
<style>
div[role="radiogroup"] { display: flex; flex-wrap: wrap; gap: 6px; }
div[role="radiogroup"] label {
    background: var(--secondary-background-color);
    padding: 6px 14px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 14px;
    border: 1px solid rgba(255,255,255,0.1);
}
div[role="radiogroup"] label:has(input:checked) {
    background: #5865f2;
    color: white;
    border-color: #5865f2;
}
div[role="radiogroup"] label input { display: none; }
</style>
""", unsafe_allow_html=True)

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

# ── Missing Values ────────────────────────────────────────────────────────────
if selected_tab == "Missing Values":
    st.subheader("Missing Values")
    if report["missing"]:
        miss_df = pd.DataFrame([
            {"Column": k, "Missing Count": v["count"], "Missing %": v["percent"]}
            for k, v in report["missing"].items()
        ])
        fig = px.bar(miss_df, x="Column", y="Missing %", color="Missing %",
                     color_continuous_scale="Reds", title="Missing Value % per Column")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(miss_df, use_container_width=True)
    else:
        st.success("No missing values found.")

# ── Outliers ──────────────────────────────────────────────────────────────────
elif selected_tab == "Outliers":
    st.subheader("Outliers (Z-score > 3)")
    if report["outliers"]:
        out_df = pd.DataFrame([
            {"Column": k, "Outlier Count": v["count"], "Outlier %": v["percent"]}
            for k, v in report["outliers"].items()
        ])
        fig = px.bar(out_df, x="Column", y="Outlier %", color="Outlier %",
                     color_continuous_scale="Oranges", title="Outlier % per Column")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out_df, use_container_width=True)
    else:
        st.success("No significant outliers detected.")

# ── Class Imbalance ───────────────────────────────────────────────────────────
elif selected_tab == "Class Imbalance":
    st.subheader("Class Imbalance")
    if report["class_imbalance"]:
        for col, info in report["class_imbalance"].items():
            st.markdown(f"**{col}** — Imbalance ratio: `{info['imbalance_ratio']}:1` {'🔴 Imbalanced' if info['is_imbalanced'] else '🟢 Acceptable'}")
            dist_df = pd.DataFrame(list(info["distribution"].items()), columns=["Class", "Count"])
            fig = px.pie(dist_df, names="Class", values="Count", title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No class imbalance issues found.")

# ── Correlations ──────────────────────────────────────────────────────────────
elif selected_tab == "Correlations":
    st.subheader("Correlation Matrix")
    matrix = report["correlations"].get("matrix", {})
    if matrix:
        corr_df = pd.DataFrame(matrix)
        fig = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Feature Correlation Heatmap", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        high = report["correlations"].get("high_correlations", [])
        if high:
            st.warning("High correlations detected (|r| > 0.8):")
            for pair in high:
                st.markdown(f"- **{pair['col1']}** ↔ **{pair['col2']}**: `{pair['correlation']}`")
    else:
        st.info("Not enough numeric columns for correlation analysis.")

# ── Column Stats ──────────────────────────────────────────────────────────────
elif selected_tab == "Column Stats":
    st.subheader("Column Statistics")
    rows = []
    for col, s in report["column_stats"].items():
        row = {"Column": col, "Type": s["dtype"], "Unique": s["unique"]}
        if "mean" in s:
            row.update({"Mean": s["mean"], "Std": s["std"], "Min": s["min"],
                        "Max": s["max"], "Skewness": s["skewness"]})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Recommendations ───────────────────────────────────────────────────────────
elif selected_tab == "Recommendations":
    st.subheader("Actionable Recommendations")
    if report["recommendations"]:
        for rec in report["recommendations"]:
            if rec.startswith("DROP"):
                st.error(f"🗑️ {rec}")
            elif rec.startswith("IMPUTE") or rec.startswith("Consider"):
                st.warning(f"⚠️ {rec}")
            elif "HIGH CORRELATION" in rec:
                st.warning(f"🔗 {rec}")
            else:
                st.info(f"💡 {rec}")
    else:
        st.success("Dataset looks clean! No major issues found.")

# ── AI Assistant ──────────────────────────────────────────────────────────────
elif selected_tab == "AI Assistant":
    st.subheader("Ask the AI about your dataset")
    st.caption("The AI has read your full EDA report. Ask anything about the analysis.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display all messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if not st.session_state["chat_history"]:
        st.info("No messages yet. Ask a question below.")

    # Input pinned at bottom
    question = st.chat_input("e.g. Why is salary an outlier? What features are highly correlated?")

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

# ── Download PDF ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Download Report")
if st.button("Generate PDF Report"):
    with st.spinner("Generating PDF..."):
        pdf_res = requests.get(f"{BACKEND}/download/{session_id}")
    if pdf_res.status_code == 200:
        st.download_button(
            label="Click to download",
            data=pdf_res.content,
            file_name=f"eda_report_{session_id}.pdf",
            mime="application/pdf"
        )
    else:
        st.error("Failed to generate PDF.")