from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Anon client — for auth operations
anon_client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Service client — for server-side DB operations (bypasses RLS safely)
service_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)