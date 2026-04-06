from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from services.eda_service import run_eda
from services.vector_service import embed_report, query_report
from services.report_service import generate_pdf
from models.schemas import ChatRequest
import io

router = APIRouter()

eda_store = {}  # temporary in-memory store keyed by session_id


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files supported")
    
    try:
        contents = await file.read()
        session_id, report = run_eda(contents, file.filename)
        eda_store[session_id] = report
        embed_report(session_id, report)
        return {"session_id": session_id, "report": report}
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full error in backend terminal
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
def chat(req: ChatRequest):
    try:
        answer = query_report(req.session_id, req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{session_id}")
def download(session_id: str):
    if session_id not in eda_store:
        raise HTTPException(status_code=404, detail="Session not found")
    pdf_bytes = generate_pdf(eda_store[session_id])
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{session_id}.pdf"}
    )