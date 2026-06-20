# DS Analyzer

**An AI-powered Exploratory Data Analysis tool** — upload a CSV or Excel file, get a full statistical report, ask questions about your data in plain English, and download a polished PDF report.

🔗 **Live Demo**: [dsanalyzer.streamlit.app](https://dsanalyzer-kausty.streamlit.app/) &nbsp;|&nbsp; **Backend API**: [ds-analyzer-backend.onrender.com](https://ds-analyzer.onrender.com)

---

## What it does

Upload any CSV or Excel dataset and DS Analyzer will:

- Detect missing values, duplicates, outliers, class imbalance, and high correlations
- Compute full column statistics (mean, std, skewness, quartiles, top values)
- Plot distributions, correlation heatmaps, and feature importance charts interactively
- Auto-detect your target column and rank features by importance using a Random Forest
- Generate actionable data-cleaning recommendations based on the analysis
- Let you chat with your dataset — ask questions in plain English, powered by Llama 3.3 70B via Groq
- Export a complete multi-section PDF report with embedded charts, ready to share

User accounts, analysis history, and saved sessions are persisted via Supabase.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (Python) |
| Backend | FastAPI + Uvicorn |
| AI / LLM | Groq API — Llama 3.3 70B Versatile |
| ML | scikit-learn (Random Forest for feature importance) |
| Data | pandas, NumPy, SciPy |
| Charts | Plotly (interactive) + Plotly Kaleido (PDF rendering) |
| PDF | fpdf2 |
| Auth & DB | Supabase (PostgreSQL + JWT auth) |
| Deployment | Render (backend) + Streamlit Cloud (frontend) |

---

## Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────────────┐
│   Streamlit Frontend        │  HTTPS  │   FastAPI Backend (Render)           │
│   (Streamlit Cloud)         │ ──────► │                                      │
│                             │         │  POST /api/analyze                   │
│  • Login / Signup           │         │    └─ run_eda()       → EDA report   │
│  • File upload              │         │    └─ embed_report()  → in-memory    │
│  • Interactive charts       │         │                                      │
│  • AI chat panel            │         │  POST /api/chat                      │
│  • Analysis history         │         │    └─ query_report()  → Groq LLM    │
│  • PDF download             │         │                                      │
└─────────────────────────────┘         │  GET  /api/download/{session_id}     │
                                        │    └─ generate_pdf() → PDF bytes     │
                                        │                                      │
                                        │  POST /auth/signup, /auth/login      │
                                        │  POST /store/save, GET /store/list   │
                                        └──────────────────────┬───────────────┘
                                                               │
                                              ┌────────────────▼──────────────┐
                                              │   Supabase                    │
                                              │   • users (auth)              │
                                              │   • profiles (username)       │
                                              │   • analyses (saved reports)  │
                                              └───────────────────────────────┘
```

The backend processes files entirely in-memory (no disk writes). EDA reports are stored in a session dict and injected directly into the LLM context — no vector database needed at this scale.

---

## EDA pipeline

For each uploaded file the backend runs:

1. **Shape & preview** — rows, columns, dtype inference, first 10 rows
2. **Missing values** — count and percentage per column
3. **Duplicates** — count and percentage across the full dataset
4. **Outlier detection** — Z-score (|z| > 3) per numeric column
5. **Class imbalance** — value distribution and imbalance ratio for categorical columns
6. **Correlation matrix** — Pearson correlation, flags pairs with |r| > 0.8
7. **Column statistics** — mean, median, std, min, max, skewness, IQR, top values
8. **Distributions** — histogram bin data (counts + edges) for all numeric columns
9. **Feature importance** — auto-detects target column by keyword matching; trains a Random Forest Classifier or Regressor (50 estimators); ranks features by importance
10. **Recommendations** — rule-based engine generates plain-English cleaning advice from all of the above

Datasets over 50,000 rows are randomly sampled before analysis.

---

## AI chat

After analysis, a chat panel lets you ask natural language questions about your data. The full EDA report is serialized into a compact context string and injected into each Groq API call alongside conversation history, so the LLM always has the full picture.

Example questions it can answer:
- *"Which features should I drop before training a model?"*
- *"Why is the salary column skewed?"*
- *"What preprocessing steps do you recommend for this dataset?"*

---

## Project structure

```
ds_analyzer/
├── backend/
│   ├── main.py                  # FastAPI app, CORS config
│   ├── dependencies.py          # JWT auth middleware
│   ├── requirements.txt
│   ├── routers/
│   │   ├── analysis.py          # /api/analyze, /api/chat, /api/download
│   │   ├── analysis_store.py    # /store/save, /store/list, /store/load, /store/delete
│   │   └── auth.py              # /auth/signup, /auth/login, /auth/verify
│   ├── services/
│   │   ├── eda_service.py       # Full EDA pipeline
│   │   ├── report_service.py    # PDF generation with embedded charts
│   │   └── vector_service.py    # Report storage + Groq LLM chat
│   ├── models/
│   │   └── schemas.py           # Pydantic request models
│   └── utils/
│       └── supabase_client.py   # Anon + service role Supabase clients
└── frontend/
    ├── app.py                   # Streamlit UI (login, analysis, chat, history)
    └── requirements.txt
```

---

## Running locally

### Prerequisites
- Python 3.11+
- A [Supabase](https://supabase.com) project with `profiles` and `analyses` tables
- A [Groq](https://groq.com) API key

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY, GROQ_API_KEY

uvicorn main:app --reload
# Runs at http://localhost:8000
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt

mkdir -p .streamlit
echo 'BACKEND_URL = "http://localhost:8000"' > .streamlit/secrets.toml

streamlit run app.py
# Runs at http://localhost:8501
```

---

## Deployment

| Service | Platform | Config file |
|---|---|---|
| Backend | [Render](https://render.com) | `render.yaml` |
| Frontend | [Streamlit Cloud](https://share.streamlit.io) | `.streamlit/secrets.toml` |

Set the following environment variables in the Render dashboard:

```
SUPABASE_URL
SUPABASE_ANON_KEY
SUPABASE_SERVICE_KEY
GROQ_API_KEY
ALLOWED_ORIGINS=https://your-app.streamlit.app
```

Set the following in Streamlit Cloud → App Settings → Secrets:

```toml
BACKEND_URL = "https://ds-analyzer-backend.onrender.com"
```

---

## Supabase schema

```sql
-- Profiles (created on signup via trigger or manually)
create table profiles (
  id uuid references auth.users primary key,
  username text
);

-- Saved analyses
create table analyses (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users not null,
  session_id text not null,
  filename text,
  report jsonb,
  created_at timestamptz default now()
);

-- RLS: users can only access their own rows
alter table analyses enable row level security;
create policy "own rows" on analyses
  using (auth.uid() = user_id);
```

---

## License

MIT
