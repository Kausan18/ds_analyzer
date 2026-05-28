from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from utils.supabase_client import service_client
from dependencies import get_current_user
import json

router = APIRouter()


class SaveRequest(BaseModel):
    session_id: str
    filename: str
    report: dict


@router.post("/save")
def save_analysis(req: SaveRequest, user=Depends(get_current_user)):
    try:
        service_client.table("analyses").insert({
            "user_id": user.id,
            "session_id": req.session_id,
            "filename": req.filename,
            "report": req.report
        }).execute()
        return {"message": "Analysis saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list/{user_id}")
def list_analyses(user_id: str, user=Depends(get_current_user)):
    try:
        res = service_client.table("analyses")\
            .select("id, session_id, filename, created_at")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        return {"analyses": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load/{session_id}/{user_id}")
def load_analysis(session_id: str, user_id: str, user=Depends(get_current_user)):
    try:
        res = service_client.table("analyses")\
            .select("*")\
            .eq("session_id", session_id)\
            .eq("user_id", user_id)\
            .single()\
            .execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        return {"report": res.data["report"], "session_id": res.data["session_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{session_id}/{user_id}")
def delete_analysis(session_id: str, user_id: str, user=Depends(get_current_user)):
    try:
        service_client.table("analyses")\
            .delete()\
            .eq("session_id", session_id)\
            .eq("user_id", user_id)\
            .execute()
        return {"message": "Deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))