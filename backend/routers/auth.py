from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from utils.supabase_client import anon_client, service_client

router = APIRouter()


class AuthRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    password: str
    username: Optional[str] = None


class TokenRequest(BaseModel):
    access_token: str


def _get_username(user_id: str) -> Optional[str]:
    """Fetch username from profiles table via service client."""
    try:
        res = (
            service_client
            .table("profiles")
            .select("username")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if res.data:
            return res.data.get("username")
    except Exception:
        pass
    return None


@router.post("/signup")
def signup(req: SignupRequest):
    try:
        res = anon_client.auth.sign_up({
            "email": req.email,
            "password": req.password,
            "options": {
                "data": {
                    "username": req.username or req.email.split("@")[0]
                }
            }
        })
        if res.user is None:
            raise HTTPException(
                status_code=400,
                detail="Signup failed. Check your email for confirmation."
            )
        return {
            "message": "Signup successful. Please check your email to confirm your account."
        }
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

        username = _get_username(res.user.id)
        if not username:
            # Fallback: use part before @ in email
            username = res.user.email.split("@")[0]

        return {
            "access_token": res.session.access_token,
            "user_id": res.user.id,
            "email": res.user.email,
            "username": username,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/verify")
def verify_token(req: TokenRequest):
    try:
        res = anon_client.auth.get_user(req.access_token)
        if res.user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")

        username = _get_username(res.user.id)
        if not username:
            username = res.user.email.split("@")[0]

        return {
            "user_id": res.user.id,
            "email": res.user.email,
            "username": username,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))