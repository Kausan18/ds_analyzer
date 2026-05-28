from fastapi import Header, HTTPException
from utils.supabase_client import anon_client


def get_current_user(authorization: str = Header(...)):
    """
    Extracts and verifies the Supabase JWT from the Authorization header.
    Returns the verified user object. Raises 401 if invalid or missing.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format.")
    token = authorization[len("Bearer "):].strip()
    try:
        res = anon_client.auth.get_user(token)
        # supabase-py returns an object with .user or similar; handle truthiness
        if getattr(res, "user", None) is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        return res.user
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed.")
