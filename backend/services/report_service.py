from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import io
import os


def safe(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def fig_to_bytes(fig, height=350) -> bytes:
    """Render plotly figure to PNG bytes in memory — no temp files."""
    return fig.to_image(format="png", width=700, height=height, scale=2)


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Dataset Evaluation Report", align="C")
        self.ln(12)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def section_title(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(235, 235, 250)
        self.set_text_color(40, 40, 100)
        self.cell(0, 8, f"  {safe(title)}", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 6, safe(text))
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 6, safe(f"  - {text}"))
        self.ln(0.5)

    def kv_row(self, label, value):
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 6, safe(f"{label}  {value}"))
        self.ln(7)

    def insert_chart(self, fig, caption: str = "", height: int = 350):
        """Render plotly fig to bytes and embed directly — no temp files."""
        img_bytes = fig_to_bytes(fig, height=height)
        page_width = self.w - self.l_margin - self.r_margin
        # fpdf2 can accept a BytesIO directly
        self.image(io.BytesIO(img_bytes), x=self.l_margin, w=page_width)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, safe(caption), align="C")
            self.set_text_color(0, 0, 0)
            self.ln(2)
        self.ln(4)


def generate_pdf(report: dict) -> bytes:
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── 1. Overview ───────────────────────────────────────────────────────────
    pdf.section_title("1. Dataset Overview")
    pdf.kv_row("Filename:", report.get("filename", "N/A"))
    pdf.kv_row("Rows:", str(report["shape"]["rows"]))
    pdf.kv_row("Columns:", str(report["shape"]["cols"]))
    if report.get("sampled"):
        pdf.kv_row("Note:", f"Sampled 50,000 rows from {report.get('total_rows_original', '?')} total")
    pdf.body_text("Columns: " + ", ".join(report["columns"]))

    # ── 2. Missing Values ─────────────────────────────────────────────────────
    pdf.section_title("2. Missing Values")
    if report["missing"]:
        for col, info in report["missing"].items():
            pdf.bullet(f"{col}: {info['count']} missing ({info['percent']}%)")

        miss_df = pd.DataFrame([
            {"Column": k, "Missing %": v["percent"]}
            for k, v in report["missing"].items()
        ])
        fig = px.bar(
            miss_df, x="Column", y="Missing %",
            color="Missing %", color_continuous_scale="Reds",
            title="Missing Value % per Column",
            template="plotly_white"
        )
        fig.update_layout(margin=dict(t=50, b=40, l=40, r=20))
        pdf.insert_chart(fig, "Fig 1: Missing values by column")
    else:
        pdf.body_text("No missing values found in any column.")

    # ── 3. Duplicates ─────────────────────────────────────────────────────────
    pdf.section_title("3. Duplicate Rows")
    dup = report["duplicates"]
    if dup["count"] > 0:
        pdf.body_text(f"{dup['count']} duplicate rows found ({dup['percent']}% of dataset). Remove before modelling.")
    else:
        pdf.body_text("No duplicate rows found.")

    # ── 4. Outliers ───────────────────────────────────────────────────────────
    pdf.section_title("4. Outliers (Z-score > 3)")
    if report["outliers"]:
        for col, info in report["outliers"].items():
            pdf.bullet(f"{col}: {info['count']} outliers ({info['percent']}%)")

        out_df = pd.DataFrame([
            {"Column": k, "Outlier %": v["percent"]}
            for k, v in report["outliers"].items()
        ])
        fig = px.bar(
            out_df, x="Column", y="Outlier %",
            color="Outlier %", color_continuous_scale="Oranges",
            title="Outlier % per Column (Z-score > 3)",
            template="plotly_white"
        )
        fig.update_layout(margin=dict(t=50, b=40, l=40, r=20))
        pdf.insert_chart(fig, "Fig 2: Outlier % by column")
    else:
        pdf.body_text("No significant outliers detected.")

    # ── 5. Class Imbalance ────────────────────────────────────────────────────
    pdf.section_title("5. Class Imbalance")
    if report["class_imbalance"]:
        for col, info in report["class_imbalance"].items():
            status = "IMBALANCED" if info["is_imbalanced"] else "Acceptable"
            pdf.bullet(f"{col}: ratio {info['imbalance_ratio']}:1  ({status})")
            dist_df = pd.DataFrame(
                list(info["distribution"].items()),
                columns=["Class", "Count"]
            )
            fig = px.pie(
                dist_df, names="Class", values="Count",
                title=f"Class Distribution — {col}",
                template="plotly_white"
            )
            fig.update_layout(margin=dict(t=50, b=20, l=20, r=20))
            pdf.insert_chart(fig, f"Fig: Distribution of '{col}'", height=300)
    else:
        pdf.body_text("No class imbalance issues detected.")

    # ── 6. Correlation Heatmap ────────────────────────────────────────────────
    pdf.section_title("6. Feature Correlation Heatmap")
    matrix = report["correlations"].get("matrix", {})
    if matrix:
        corr_df = pd.DataFrame(matrix)
        n_cols = len(corr_df.columns)
        heatmap_height = max(350, min(n_cols * 40, 650))

        fig = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
            zmin=-1, zmax=1,
            template="plotly_white"
        )
        fig.update_layout(margin=dict(t=50, b=40, l=60, r=20))
        pdf.insert_chart(fig, "Fig 3: Red = positive correlation, Blue = negative", height=heatmap_height)

        pairs = report["correlations"].get("high_correlations", [])
        if pairs:
            pdf.body_text("Highly correlated pairs (|r| > 0.8):")
            for p in pairs:
                pdf.bullet(f"{p['col1']}  <->  {p['col2']}:  r = {p['correlation']}")
        else:
            pdf.body_text("No feature pairs exceed the |r| > 0.8 threshold.")
    else:
        pdf.body_text("Not enough numeric columns for correlation analysis.")

    # ── 7. Column Statistics ──────────────────────────────────────────────────
    pdf.section_title("7. Column Statistics")
    numeric_rows = []
    for col, s in report["column_stats"].items():
        if "mean" in s:
            numeric_rows.append({
                "Column": col, "Mean": s["mean"], "Std": s["std"],
                "Min": s["min"], "Max": s["max"],
                "Skewness": s["skewness"], "Unique": s["unique"]
            })

    if numeric_rows:
        skew_df = pd.DataFrame(numeric_rows)
        fig = px.bar(
            skew_df, x="Column", y="Skewness",
            color="Skewness",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            title="Skewness per Numeric Column",
            template="plotly_white"
        )
        fig.update_layout(margin=dict(t=50, b=40, l=40, r=20))
        pdf.insert_chart(fig, "Fig 4: Skewness per column (|skew| > 1 suggests log transform)")

        pdf.body_text("Numeric column summary:")
        for row in numeric_rows:
            pdf.bullet(
                f"{row['Column']}: mean={row['Mean']}, std={row['Std']}, "
                f"min={row['Min']}, max={row['Max']}, "
                f"skew={row['Skewness']}, unique={row['Unique']}"
            )

    for col, s in report["column_stats"].items():
        if "top_values" in s:
            top = ", ".join([f"{k}({v})" for k, v in list(s["top_values"].items())[:5]])
            pdf.bullet(f"{col} (categorical): {s['unique']} unique — top: {top}")

# ── 7b. Feature Importance ────────────────────────────────────────────────
    pdf.section_title("7b. Feature Importance")
    fi = report.get("feature_importance", {})
    if fi.get("available"):
        pdf.body_text(f"Target column: '{fi['target_column']}' | Model: Random Forest {fi['model_type'].title()}")
        fi_df = pd.DataFrame(fi["features"])
        fig = px.bar(
            fi_df, x="importance", y="feature",
            orientation="h",
            title=f"Feature Importance for '{fi['target_column']}'",
            template="plotly_white",
            color="importance", color_continuous_scale="Greens"
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            margin=dict(t=50, b=40, l=120, r=20),
            height=max(350, len(fi_df) * 25)
        )
        pdf.insert_chart(fig, "Fig 5: Feature importance scores", height=max(350, len(fi_df) * 25))
        pdf.body_text("Top features:")
        for f in fi["features"][:10]:
            pdf.bullet(f"{f['feature']}: {f['importance']}")
    else:
        pdf.body_text(f"Not available: {fi.get('reason', 'unknown')}")

    # ── 7c. Distributions ─────────────────────────────────────────────────────
    pdf.section_title("7c. Column Distributions")
    distributions = report.get("distributions", {})
    if distributions:
        dist_items = list(distributions.items())
        for i in range(0, min(len(dist_items), 8), 1):
            col_name, d = dist_items[i]
            s = report["column_stats"].get(col_name, {})
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=d["bin_centers"], y=d["counts"],
                name=col_name, marker_color="steelblue"
            ))
            if "mean" in s:
                fig.add_vline(x=s["mean"], line_dash="dash",
                              line_color="red", annotation_text=f"Mean={s['mean']}")
            fig.update_layout(
                title=f"Distribution of '{col_name}'",
                template="plotly_white",
                height=280,
                margin=dict(t=40, b=30, l=40, r=20)
            )
            pdf.insert_chart(fig, f"Skewness={s.get('skewness', 'N/A')}", height=280)
    else:
        pdf.body_text("No numeric distributions available.")

    # ── 8. Recommendations ───────────────────────────────────────────────────
    pdf.section_title("8. Actionable Recommendations")
    if report["recommendations"]:
        for rec in report["recommendations"]:
            pdf.bullet(rec)
    else:
        pdf.body_text("Dataset looks clean. No major issues found.")

    return bytes(pdf.output())