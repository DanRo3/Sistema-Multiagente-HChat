import google.generativeai as genai
from app.core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

_llm_client = None

def configure_genai():
    """Configura la API de Google GenAI."""
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        print("Google GenAI configurado.")
        return True
    except Exception as e:
        print(f"Error configurando Google GenAI: {e}")
        return False

def get_gemini_llm(temperature: float = 0.1, model_name: str = None) -> Optional[ChatGoogleGenerativeAI]:
    """
    Obtiene una instancia del modelo LLM de Gemini configurado.
    Usa un patrón singleton simple.
    """
    global _llm_client
    if not model_name:
        model_name = settings.GEMINI_MODEL_NAME

    # Podríamos hacer el singleton más robusto, pero para esto sirve
    # if _llm_client and _llm_client.model_name == model_name and _llm_client.temperature == temperature:
    #     return _llm_client

    if not configure_genai(): # Asegura que esté configurado
         return None

    try:
        print(f"Inicializando LLM: {model_name} con temperatura {temperature}")
        _llm_client = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            # Puedes añadir más configuraciones aquí (top_p, top_k, safety_settings)
            # convert_system_message_to_human=True # A veces útil para compatibilidad
        )
        return _llm_client
    except Exception as e:
        print(f"Error al inicializar ChatGoogleGenerativeAI: {e}")
        _llm_client = None
        return None

# Ejemplo de uso (esto iría dentro de los agentes)
# llm = get_gemini_llm()
# if llm:
#     response = llm.invoke("Tu prompt aquí")
#     print(response.content)