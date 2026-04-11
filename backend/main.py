from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analysis, auth, analysis_store

app = FastAPI(title="Dataset Evaluator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.streamlit.app",  # add after Streamlit deploy
        "http://localhost:8501",   
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api")
app.include_router(auth.router, prefix="/auth")
app.include_router(analysis_store.router, prefix="/store")

@app.get("/ping")
def ping():
    return {"message": "pong"}