import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gobj
import os
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Dataset Quality Evaluator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "authenticated": False,
    "access_token": None,
    "user_id": None,
    "user_email": None,
    "page": "login",
    "report": None,
    "session_id": None,
    "chat_history": [],
    "active_tab": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────
def api(method, path, **kwargs):
    url = f"{BACKEND}{path}"
    try:
        res = getattr(requests, method)(url, timeout=300, **kwargs)
        return res
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach backend. Is uvicorn running?")
        return None


def logout():
    for key in ["authenticated", "access_token", "user_id", "user_email",
                "report", "session_id", "chat_history", "active_tab"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["page"] = "login"
    st.session_state["authenticated"] = False
    st.rerun()


def go(page):
    st.session_state["page"] = page
    st.session_state["active_tab"] = 0
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN / SIGNUP
# ════════════════════════════════════════════════════════════════════════════
def page_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## Dataset Quality Evaluator")
        st.markdown("##### AI-powered EDA and data quality analysis")
        st.divider()

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", type="primary", use_container_width=True):
                if not email or not password:
                    st.error("Please enter email and password.")
                else:
                    res = api("post", "/auth/login",
                              json={"email": email, "password": password})
                    if res and res.status_code == 200:
                        data = res.json()
                        st.session_state["authenticated"] = True
                        st.session_state["access_token"] = data["access_token"]
                        st.session_state["user_id"] = data["user_id"]
                        st.session_state["user_email"] = data["email"]
                        go("dashboard")
                    else:
                        detail = res.json().get("detail", "Login failed.") if res else "No response."
                        st.error(detail)

        with tab_signup:
            email_s = st.text_input("Email", key="signup_email")
            password_s = st.text_input("Password (min 6 chars)", type="password", key="signup_pass")
            password_s2 = st.text_input("Confirm Password", type="password", key="signup_pass2")

            if st.button("Create Account", type="primary", use_container_width=True):
                if not email_s or not password_s:
                    st.error("Please fill all fields.")
                elif password_s != password_s2:
                    st.error("Passwords do not match.")
                elif len(password_s) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    res = api("post", "/auth/signup",
                              json={"email": email_s, "password": password_s})
                    if res and res.status_code == 200:
                        st.success("Account created! Please check your email to confirm, then log in.")
                    else:
                        detail = res.json().get("detail", "Signup failed.") if res else "No response."
                        st.error(detail)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"## Welcome back, {st.session_state['user_email'].split('@')[0]} 👋")
        st.caption("What would you like to do today?")
    with col2:
        if st.button("Logout", use_container_width=True):
            logout()

    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div style="border:1px solid rgba(255,255,255,0.15); border-radius:12px; padding:32px; text-align:center;">
            <div style="font-size:48px">📊</div>
            <h3>New Analysis</h3>
            <p style="color:#aaa">Upload a CSV or Excel dataset and get a full automated EDA report with AI insights.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Start New Analysis", type="primary", use_container_width=True):
            st.session_state["report"] = None
            st.session_state["session_id"] = None
            st.session_state["chat_history"] = []
            st.session_state["active_tab"] = 0
            go("analysis")

    with c2:
        st.markdown("""
        <div style="border:1px solid rgba(255,255,255,0.15); border-radius:12px; padding:32px; text-align:center;">
            <div style="font-size:48px">📁</div>
            <h3>Past Analyses</h3>
            <p style="color:#aaa">View and revisit your previously saved EDA reports and AI conversations.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("View Past Analyses", use_container_width=True):
            go("past")

    st.divider()
    st.markdown("### Recent Analyses")
    res = api("get", f"/store/list/{st.session_state['user_id']}")
    if res and res.status_code == 200:
        analyses = res.json().get("analyses", [])
        if analyses:
            for item in analyses[:5]:
                col_a, col_b, col_c = st.columns([3, 2, 1])
                with col_a:
                    st.markdown(f"**{item['filename']}**")
                with col_b:
                    created = item["created_at"][:19].replace("T", " ")
                    st.caption(created)
                with col_c:
                    if st.button("Open", key=f"open_{item['session_id']}"):
                        with st.spinner("Loading..."):
                            load_past_analysis(item["session_id"])
        else:
            st.info("No analyses yet. Start your first one above.")
    else:
        st.info("Could not load recent analyses.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PAST ANALYSES
# ════════════════════════════════════════════════════════════════════════════
def load_past_analysis(session_id: str):
    res = api("get", f"/store/load/{session_id}/{st.session_state['user_id']}")
    if res and res.status_code == 200:
        data = res.json()
        st.session_state["report"] = data["report"]
        st.session_state["session_id"] = data["session_id"]
        st.session_state["chat_history"] = []
        st.session_state["active_tab"] = 0
        api("post", "/api/reembed",
            json={"session_id": data["session_id"], "report": data["report"]})
        go("analysis")
    else:
        st.error("Could not load analysis.")


def page_past():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("## Past Analyses")
    with col2:
        if st.button("Back to Dashboard"):
            go("dashboard")

    st.divider()

    res = api("get", f"/store/list/{st.session_state['user_id']}")
    if not res or res.status_code != 200:
        st.error("Could not load analyses.")
        return

    analyses = res.json().get("analyses", [])
    if not analyses:
        st.info("No saved analyses found.")
        return

    for item in analyses:
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            with c1:
                st.markdown(f"**{item['filename']}**")
                st.caption(f"Session: {item['session_id']}")
            with c2:
                created = item["created_at"][:19].replace("T", " ")
                st.caption(f"Created: {created}")
            with c3:
                if st.button("Open", key=f"past_open_{item['session_id']}", use_container_width=True):
                    with st.spinner("Loading..."):
                        load_past_analysis(item["session_id"])
            with c4:
                if st.button("Delete", key=f"del_{item['session_id']}", use_container_width=True):
                    del_res = api("delete",
                                  f"/store/delete/{item['session_id']}/{st.session_state['user_id']}")
                    if del_res and del_res.status_code == 200:
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.error("Delete failed.")
            st.divider()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSIS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
def page_analysis():
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown("## Dataset Quality Evaluator")
    with col2:
        if st.button("Dashboard", use_container_width=True):
            go("dashboard")
    with col3:
        if st.button("Logout", use_container_width=True):
            logout()

    # ── Upload ────────────────────────────────────────────────────────────────
    if not st.session_state["report"]:
        st.divider()
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV or Excel)",
            type=["csv", "xlsx", "xls"]
        )
        if uploaded_file:
            progress = st.progress(0, text="Reading file...")
            try:
                progress.progress(20, text="Sending to analysis engine...")
                res = api("post", "/api/analyze",
                          files={"file": (uploaded_file.name,
                                          uploaded_file.getvalue(),
                                          uploaded_file.type)})
                progress.progress(80, text="Processing results...")

                if res and res.status_code == 200:
                    data = res.json()
                    st.session_state["report"] = data["report"]
                    st.session_state["session_id"] = data["session_id"]
                    st.session_state["chat_history"] = []

                    api("post", "/store/save", json={
                        "user_id": st.session_state["user_id"],
                        "session_id": data["session_id"],
                        "filename": uploaded_file.name,
                        "report": data["report"]
                    })

                    progress.progress(100, text="Done!")
                    if data["report"].get("sampled"):
                        st.warning(
                            f"Large dataset — {data['report']['total_rows_original']:,} rows. "
                            f"Analysis on 50,000-row sample."
                        )
                    st.success("Analysis complete and saved!")
                    st.rerun()
                else:
                    progress.empty()
                    detail = res.json().get("detail", "Unknown") if res else "No response"
                    st.error(f"Error: {detail}")
            except Exception as e:
                progress.empty()
                st.error(str(e))
        return

    # ── Report loaded ─────────────────────────────────────────────────────────
    report = st.session_state["report"]
    session_id = st.session_state["session_id"]

    st.divider()

    # Metric cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Rows", f"{report['shape']['rows']:,}")
    with c2:
        st.metric("Columns", report["shape"]["cols"])
    with c3:
        st.metric("Missing Cols", len(report["missing"]))
    with c4:
        st.metric("Duplicates", report["duplicates"]["count"])
    with c5:
        st.metric("Outlier Cols", len(report["outliers"]))
    with c6:
        fi = report.get("feature_importance", {})
        st.metric("Target Col", fi.get("target_column", "N/A") if fi.get("available") else "N/A")

    st.divider()

    # Tab nav
    tab_names = [
        "Preview", "Missing Values", "Outliers", "Distributions",
        "Class Imbalance", "Correlations", "Feature Importance",
        "Column Stats", "Recommendations", "AI Assistant"
    ]
    selected_tab = st.radio(
        label="", options=tab_names,
        index=st.session_state["active_tab"],
        horizontal=True, key="tab_selector",
        label_visibility="collapsed"
    )
    st.session_state["active_tab"] = tab_names.index(selected_tab)
    st.divider()

    # ── Preview ───────────────────────────────────────────────────────────────
    if selected_tab == "Preview":
        st.subheader("Dataset Preview")
        preview = report.get("dataset_preview", {})
        if preview:
            preview_df = pd.DataFrame(preview["rows"], columns=preview["columns"])
            st.dataframe(preview_df, use_container_width=True)
            st.caption(f"First 10 rows of {report['shape']['rows']:,} total rows.")
        dtype_df = pd.DataFrame([
            {"Column": col, "Type": dtype,
             "Category": "Numeric" if "int" in dtype or "float" in dtype
             else "Categorical" if dtype == "object" else "Other"}
            for col, dtype in report["dtypes"].items()
        ])
        fig = px.pie(dtype_df, names="Category",
                     title="Column Type Distribution", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dtype_df, use_container_width=True)

    # ── Missing Values ────────────────────────────────────────────────────────
    elif selected_tab == "Missing Values":
        st.subheader("Missing Values")
        if report["missing"]:
            miss_df = pd.DataFrame([
                {"Column": k, "Missing Count": v["count"], "Missing %": v["percent"]}
                for k, v in report["missing"].items()
            ]).sort_values("Missing %", ascending=False)
            fig = px.bar(miss_df, x="Column", y="Missing %", color="Missing %",
                         color_continuous_scale="Reds",
                         title="Missing Value % per Column", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss_df, use_container_width=True)
        else:
            st.success("No missing values found.")

    # ── Outliers ──────────────────────────────────────────────────────────────
    elif selected_tab == "Outliers":
        st.subheader("Outliers (Z-score > 3)")
        if report["outliers"]:
            out_df = pd.DataFrame([
                {"Column": k, "Outlier Count": v["count"], "Outlier %": v["percent"]}
                for k, v in report["outliers"].items()
            ]).sort_values("Outlier %", ascending=False)
            fig = px.bar(out_df, x="Column", y="Outlier %", color="Outlier %",
                         color_continuous_scale="Oranges",
                         title="Outlier % per Column", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out_df, use_container_width=True)
        else:
            st.success("No significant outliers detected.")

    # ── Distributions ─────────────────────────────────────────────────────────
    elif selected_tab == "Distributions":
        st.subheader("Column Distributions")
        distributions = report.get("distributions", {})
        if distributions:
            cols_list = list(distributions.keys())
            selected_cols = st.multiselect(
                "Select columns", options=cols_list, default=cols_list[:6])
            if not selected_cols:
                selected_cols = cols_list[:6]

            for row_i in range(0, len(selected_cols), 2):
                cols = st.columns(2)
                for col_i in range(2):
                    idx = row_i + col_i
                    if idx >= len(selected_cols):
                        break
                    col_name = selected_cols[idx]
                    d = distributions[col_name]
                    s = report["column_stats"].get(col_name, {})
                    with cols[col_i]:
                        fig = gobj.Figure()
                        fig.add_trace(gobj.Bar(
                            x=d["bin_centers"], y=d["counts"],
                            marker_color="steelblue"
                        ))
                        if "mean" in s:
                            fig.add_vline(
                                x=s["mean"], line_dash="dash", line_color="red",
                                annotation_text=f"Mean: {s['mean']}"
                            )
                        fig.update_layout(
                            title=col_name, template="plotly_white",
                            height=280, showlegend=False,
                            margin=dict(t=40, b=30, l=30, r=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        if "skewness" in s:
                            skew = s["skewness"]
                            if abs(skew) > 1:
                                st.warning(f"Highly skewed: {skew}")
                            elif abs(skew) > 0.5:
                                st.info(f"Moderately skewed: {skew}")
                            else:
                                st.success(f"Normal-ish: {skew}")
        else:
            st.info("No numeric columns found.")

    # ── Class Imbalance ───────────────────────────────────────────────────────
    elif selected_tab == "Class Imbalance":
        st.subheader("Class Imbalance")
        if report["class_imbalance"]:
            for col, info in report["class_imbalance"].items():
                badge = "🔴 Imbalanced" if info["is_imbalanced"] else "🟢 Acceptable"
                st.markdown(f"**{col}** — Ratio: `{info['imbalance_ratio']}:1`  {badge}")
                dist_df = pd.DataFrame(
                    list(info["distribution"].items()), columns=["Class", "Count"])
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.pie(dist_df, names="Class", values="Count",
                                 title=f"Distribution of {col}", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.bar(dist_df, x="Class", y="Count",
                                 title=f"Counts — {col}", template="plotly_white",
                                 color="Count", color_continuous_scale="Blues")
                    st.plotly_chart(fig, use_container_width=True)
                st.divider()
        else:
            st.success("No class imbalance issues found.")

    # ── Correlations ──────────────────────────────────────────────────────────
    elif selected_tab == "Correlations":
        st.subheader("Correlation Matrix")
        matrix = report["correlations"].get("matrix", {})
        if matrix:
            corr_df = pd.DataFrame(matrix)
            fig = px.imshow(
                corr_df, text_auto=True, color_continuous_scale="RdBu_r",
                title="Feature Correlation Heatmap", zmin=-1, zmax=1,
                template="plotly_white"
            )
            fig.update_layout(height=max(400, len(corr_df.columns) * 40))
            st.plotly_chart(fig, use_container_width=True)
            high = report["correlations"].get("high_correlations", [])
            if high:
                st.warning(f"**{len(high)} highly correlated pairs (|r| > 0.8):**")
                st.dataframe(pd.DataFrame(high), use_container_width=True)
            else:
                st.success("No high correlations found.")
        else:
            st.info("Not enough numeric columns.")

    # ── Feature Importance ────────────────────────────────────────────────────
    elif selected_tab == "Feature Importance":
        st.subheader("Feature Importance")
        fi = report.get("feature_importance", {})
        if fi.get("available"):
            st.info(f"Target: **{fi['target_column']}** | Model: Random Forest {fi['model_type'].title()}")
            fi_df = pd.DataFrame(fi["features"])
            fi_df["signal"] = fi_df["importance"].apply(
                lambda x: "High" if x > 0.1 else ("Medium" if x > 0.03 else "Low"))
            fig = px.bar(
                fi_df, x="importance", y="feature", orientation="h",
                color="signal",
                color_discrete_map={"High": "#4ade80", "Medium": "#fbbf24", "Low": "#f87171"},
                title=f"Feature Importance → '{fi['target_column']}'",
                template="plotly_white"
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=max(400, len(fi_df) * 30)
            )
            st.plotly_chart(fig, use_container_width=True)
            fi_df["recommendation"] = fi_df["importance"].apply(
                lambda x: "Keep — high signal" if x > 0.1
                else ("Keep — moderate" if x > 0.03 else "Consider dropping"))
            st.dataframe(fi_df[["feature", "importance", "recommendation"]],
                         use_container_width=True)
        else:
            st.warning(f"Not available: {fi.get('reason', 'unknown')}")

    # ── Column Stats ──────────────────────────────────────────────────────────
    elif selected_tab == "Column Stats":
        st.subheader("Column Statistics")
        numeric_rows, cat_rows = [], []
        for col, s in report["column_stats"].items():
            if "mean" in s:
                numeric_rows.append({
                    "Column": col, "Mean": s["mean"], "Median": s["median"],
                    "Std": s["std"], "Min": s["min"], "Max": s["max"],
                    "Skewness": s["skewness"], "Q25": s.get("q25", ""),
                    "Q75": s.get("q75", ""), "Unique": s["unique"]
                })
            else:
                top = ", ".join([f"{k}({v})"
                                 for k, v in list(s.get("top_values", {}).items())[:3]])
                cat_rows.append({"Column": col, "Unique": s["unique"], "Top Values": top})
        if numeric_rows:
            st.markdown("**Numeric Columns**")
            st.dataframe(pd.DataFrame(numeric_rows), use_container_width=True)
            skew_df = pd.DataFrame(numeric_rows)[["Column", "Skewness"]]
            fig = px.bar(skew_df, x="Column", y="Skewness", color="Skewness",
                         color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                         title="Skewness per Column", template="plotly_white")
            fig.add_hline(y=1, line_dash="dash", line_color="orange")
            fig.add_hline(y=-1, line_dash="dash", line_color="orange")
            st.plotly_chart(fig, use_container_width=True)
        if cat_rows:
            st.markdown("**Categorical Columns**")
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    elif selected_tab == "Recommendations":
        st.subheader("Actionable Recommendations")
        if report["recommendations"]:
            for i, rec in enumerate(report["recommendations"]):
                if rec.startswith("DROP"):
                    st.error(f"🗑️ **{i+1}.** {rec}")
                elif any(w in rec for w in ["IMPUTE", "Consider", "HIGH CORR"]):
                    st.warning(f"⚠️ **{i+1}.** {rec}")
                elif "skewed" in rec.lower():
                    st.info(f"📊 **{i+1}.** {rec}")
                elif "important" in rec.lower():
                    st.success(f"🎯 **{i+1}.** {rec}")
                else:
                    st.info(f"💡 **{i+1}.** {rec}")
        else:
            st.success("Dataset looks clean!")

    # ── AI Assistant ──────────────────────────────────────────────────────────
    elif selected_tab == "AI Assistant":
        st.subheader("Ask the AI about your dataset")
        st.caption("The AI has full context of your EDA report.")

        if not st.session_state["chat_history"]:
            st.markdown("**Suggested questions:**")
            suggestions = [
                "Summarize all issues in this dataset",
                "Which features should I drop?",
                "Explain the correlation heatmap",
                "Which columns need transformation?",
                "What is the most important feature?"
            ]
            cols = st.columns(len(suggestions))
            for i, s in enumerate(suggestions):
                with cols[i]:
                    if st.button(s, key=f"sugg_{i}", use_container_width=True):
                        st.session_state["chat_history"].append(
                            {"role": "user", "content": s})
                        with st.spinner("Thinking..."):
                            res = api("post", "/api/chat",
                                      json={"session_id": session_id, "question": s})
                            answer = res.json()["answer"] if res and res.status_code == 200 else "Error"
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": answer})
                        st.rerun()
            st.divider()

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask anything about your dataset...")
        if question:
            st.session_state["chat_history"].append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                res = api("post", "/api/chat",
                          json={"session_id": session_id, "question": question})
                answer = res.json()["answer"] if res and res.status_code == 200 else "Error reaching AI."
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("Clear chat"):
                st.session_state["chat_history"] = []
                st.rerun()

    # ── Download PDF ──────────────────────────────────────────────────────────
    st.divider()
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF..."):
                pdf_res = requests.get(f"{BACKEND}/api/download/{session_id}")
            if pdf_res.status_code == 200:
                st.download_button(
                    "Download PDF", data=pdf_res.content,
                    file_name=f"eda_report_{session_id}.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Failed to generate PDF.")
    with c2:
        if st.button("Start New Analysis"):
            st.session_state["report"] = None
            st.session_state["session_id"] = None
            st.session_state["chat_history"] = []
            st.session_state["active_tab"] = 0
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state["authenticated"]:
    page_login()
else:
    page = st.session_state["page"]
    if page == "dashboard":
        page_dashboard()
    elif page == "analysis":
        page_analysis()
    elif page == "past":
        page_past()
    else:
        page_login()