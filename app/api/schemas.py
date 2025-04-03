# app/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    """
    Modelo para la solicitud de consulta del usuario.
    """
    query: str = Field(..., description="La consulta del usuario en lenguaje natural.", min_length=3)
    # Podríamos añadir otros campos opcionales aquí si fueran necesarios,
    # como user_id, session_id, o parámetros específicos de búsqueda.

class QueryResponse(BaseModel):
    """
    Modelo para la respuesta del sistema multiagente.
    Puede contener texto, una imagen (codificada en base64), o un mensaje de error.
    """
    text_response: Optional[str] = Field(None, description="La respuesta textual generada por el sistema.")
    image_response: Optional[str] = Field(None, description="La imagen generada codificada en Base64 con prefijo Data URI (si aplica).")
    error: Optional[str] = Field(None, description="Mensaje de error si ocurrió un problema durante el procesamiento.")

    # Ejemplo de cómo podría verse una respuesta exitosa con texto:
    # { "text_response": "El capitán Litlejohn comandó el Charles Edwin.", "image_response": null, "error": null }
    # Ejemplo de cómo podría verse una respuesta exitosa con imagen:
    # { "text_response": "Aquí tienes la visualización generada:", "image_response": "data:image/png;base64,...", "error": null }
    # Ejemplo de cómo podría verse una respuesta con error:
    # { "text_response": null, "image_response": null, "error": "Lo siento, ocurrió un error..." }