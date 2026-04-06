from fpdf import FPDF


def safe(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


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
        """Safe key-value row using full width multi_cell."""
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 6, safe(f"{label}  {value}"))
        self.ln(7)


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
    else:
        pdf.body_text("No missing values found.")

    # ── 3. Duplicates ─────────────────────────────────────────────────────────
    pdf.section_title("3. Duplicate Rows")
    dup = report["duplicates"]
    if dup["count"] > 0:
        pdf.body_text(f"{dup['count']} duplicate rows ({dup['percent']}% of dataset).")
    else:
        pdf.body_text("No duplicate rows found.")

    # ── 4. Outliers ───────────────────────────────────────────────────────────
    pdf.section_title("4. Outliers  (Z-score > 3)")
    if report["outliers"]:
        for col, info in report["outliers"].items():
            pdf.bullet(f"{col}: {info['count']} outliers ({info['percent']}%)")
    else:
        pdf.body_text("No significant outliers detected.")

    # ── 5. Class Imbalance ────────────────────────────────────────────────────
    pdf.section_title("5. Class Imbalance")
    if report["class_imbalance"]:
        for col, info in report["class_imbalance"].items():
            status = "IMBALANCED" if info["is_imbalanced"] else "Acceptable"
            pdf.bullet(f"{col}: ratio {info['imbalance_ratio']}:1  ({status})")
            dist_str = ", ".join([f"{k}: {v}" for k, v in list(info["distribution"].items())[:6]])
            pdf.bullet(f"    Distribution: {dist_str}")
    else:
        pdf.body_text("No class imbalance issues detected.")

    # ── 6. Correlations ───────────────────────────────────────────────────────
    pdf.section_title("6. High Correlations  (|r| > 0.8)")
    pairs = report["correlations"].get("high_correlations", [])
    if pairs:
        for p in pairs:
            pdf.bullet(f"{p['col1']}  <->  {p['col2']}:  r = {p['correlation']}")
    else:
        pdf.body_text("No high correlations found.")

    # ── 7. Column Stats ───────────────────────────────────────────────────────
    pdf.section_title("7. Column Statistics")
    for col, s in report["column_stats"].items():
        if "mean" in s:
            line = (f"{col}  |  mean={s['mean']}, std={s['std']}, "
                    f"min={s['min']}, max={s['max']}, skew={s['skewness']}, unique={s['unique']}")
        else:
            top = ", ".join([f"{k}({v})" for k, v in list(s.get("top_values", {}).items())[:4]])
            line = f"{col}  |  type={s['dtype']}, unique={s['unique']},  top: {top}"
        pdf.bullet(line)

    # ── 8. Recommendations ───────────────────────────────────────────────────
    pdf.section_title("8. Actionable Recommendations")
    if report["recommendations"]:
        for rec in report["recommendations"]:
            pdf.bullet(rec)
    else:
        pdf.body_text("Dataset looks clean. No major issues found.")

    return bytes(pdf.output())