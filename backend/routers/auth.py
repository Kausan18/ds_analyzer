from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.supabase_client import anon_client, service_client

router = APIRouter()


class AuthRequest(BaseModel):
    email: str
    password: str


class TokenRequest(BaseModel):
    access_token: str


@router.post("/signup")
def signup(req: AuthRequest):
    try:
        res = anon_client.auth.sign_up({
            "email": req.email,
            "password": req.password
        })
        if res.user is None:
            raise HTTPException(status_code=400, detail="Signup failed. Check your email for confirmation.")
        return {"message": "Signup successful. Please check your email to confirm your account."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
def login(req: AuthRequest):
    try:
        res = anon_client.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password
        })
        if res.user is None:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        return {
            "access_token": res.session.access_token,
            "user_id": res.user.id,
            "email": res.user.email
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/verify")
def verify_token(req: TokenRequest):
    try:
        res = anon_client.auth.get_user(req.access_token)
        if res.user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        return {"user_id": res.user.id, "email": res.user.email}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))