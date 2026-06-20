from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from routers import analysis, auth, analysis_store

app = FastAPI(title="Dataset Evaluator")

# Build allowed origins from env so you never need to redeploy to add a URL.
# In the Render dashboard set:
#   ALLOWED_ORIGINS=https://your-streamlit-app.streamlit.app,https://ds-analyzer-frontend.onrender.com
_raw = os.getenv("ALLOWED_ORIGINS", "")
_extra = [o.strip() for o in _raw.split(",") if o.strip()]

ALLOWED_ORIGINS = [
    "http://localhost:8501",   # local Streamlit dev
    "http://localhost:3000",   # any local frontend dev
    *_extra,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api")
app.include_router(auth.router, prefix="/auth")
app.include_router(analysis_store.router, prefix="/store")


@app.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    return {"message": "pong"}