# app/core/llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import settings

# Variable global para cachear el cliente LLM (opcional)
_llm_client = None

def get_llm():
    """
    Configura y devuelve el cliente del LLM (Gemini).
    Cachea el cliente para reutilizar la conexión.
    """
    global _llm_client
    if _llm_client is None:
        print(f"Initializing LLM: {settings.generative_model_name}")
        if settings.gemini_api_key == "NO_KEY_LOADED":
             raise ValueError("GEMINI_API_KEY not found in .env file.")
        try:
            _llm_client = ChatGoogleGenerativeAI(
                model=settings.generative_model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0.7, # Ajusta según necesites
                # Puedes añadir más parámetros aquí (top_p, top_k, etc.)
                convert_system_message_to_human=True # A veces necesario para compatibilidad
            )
            print("LLM client initialized.")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
    return _llm_client

# Puedes añadir funciones helper aquí si quieres estandarizar llamadas al LLM
# async def generate_text(prompt: str) -> str:
#     llm = get_llm()
#     response = await llm.ainvoke(prompt)
#     return response.content

# Ejemplo de uso (puedes quitar esto después)
# if __name__ == "__main__":
#     try:
#         llm = get_llm()
#         # Uso síncrono simple para probar
#         # Nota: LangChain v0.1+ prefiere .invoke() o .ainvoke()
#         from langchain_core.messages import HumanMessage
#         message = HumanMessage(content="Hola Gemini, ¿cómo estás?")
#         response = llm.invoke([message])
#         print("LLM Response:")
#         print(response.content)
#     except Exception as e:
#         print(f"Failed to test LLM: {e}")