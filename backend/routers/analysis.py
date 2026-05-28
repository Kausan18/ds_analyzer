from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from services.eda_service import run_eda
from services.vector_service import embed_report, query_report, report_store
from services.report_service import generate_pdf
from models.schemas import ChatRequest, ReembedRequest
from dependencies import get_current_user
import io

router = APIRouter()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files supported")
    
    try:
        MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum allowed size is 50MB."
            )
        session_id, report = run_eda(contents, file.filename)
        embed_report(session_id, report)  # stores into report_store internally
        return {"session_id": session_id, "report": report}
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full error in backend terminal
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
def chat(req: ChatRequest, user=Depends(get_current_user)):
    try:
        history = [msg.dict() for msg in req.history] if req.history else []
        answer = query_report(req.session_id, req.question, history=history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{session_id}")
def download(session_id: str, user=Depends(get_current_user)):
    if session_id not in report_store:
        raise HTTPException(status_code=404, detail="Session not found")
    pdf_bytes = generate_pdf(report_store[session_id])
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{session_id}.pdf"}
    )

@router.post("/reembed")
def reembed(req: ReembedRequest):
    try:
        embed_report(req.session_id, req.report)
        return {"message": "Report re-embedded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))