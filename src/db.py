import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def get_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Thiếu SUPABASE_URL hoặc SUPABASE_KEY trong file .env")
    return create_client(url, key)


def read_table(table_name, limit=10000):
    client = get_client()
    response = client.table(table_name).select("*").limit(limit).execute()
    return response.data
