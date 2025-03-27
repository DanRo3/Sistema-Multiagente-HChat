# app/api/schemas.py
from pydantic import BaseModel
from typing import Optional, Any

class QueryRequest(BaseModel):
    query: str
    # Puedes añadir más parámetros si los necesitas, ej: user_id, session_id

class QueryResponse(BaseModel):
    text_response: Optional[str] = None
    image_response: Optional[str] = None # Base64 encoded image string
    # Considera añadir más info: fuentes consultadas, confianza, etc.
    debug_info: Optional[dict] = None # Para depuración
    error: Optional[str] = None