import json
import re
from typing import Dict, Optional, Any
from app.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate 

# Definición del prompt para el agente moderador
MODERATOR_PROMPT = """**Tarea:** Analiza la consulta del usuario y extrae información estructurada para guiar la búsqueda y respuesta en un sistema de consulta de registros marítimos históricos.

    **Consulta del Usuario:**
    "{{query}}"

    **Contexto de Datos:** La base de datos contiene registros sobre viajes marítimos con las siguientes columnas de metadatos relevantes para filtrar: `publication_date`, `news_section`, `travel_departure_date`, `travel_duration`, `travel_arrival_date`, `travel_departure_port`, `travel_port_of_call_list`, `travel_arrival_port`, `ship_type`, `ship_name`, `cargo_list`, `master_role`, `master_name`. La columna `parsed_text` contiene el texto completo para búsqueda vectorial.

    **Instrucciones Detalladas:**
    1.  **Determina la Intención Principal (`intent`):** Clasifica como 'visual' si pide gráficos/tablas/visualizaciones, 'code' si pide código Python explícitamente, o 'text' en los demás casos.
    2.  **Identifica Filtros de Metadatos (`filters`):** Busca valores específicos en la consulta que coincidan con las columnas de metadatos listadas. Devuelve un diccionario Python con `{columna: valor}` o `null` si no hay filtros. NO inventes filtros.
    3.  **Crea una Consulta de Búsqueda Vectorial (`search_query`):** Formula una consulta concisa con palabras clave y entidades (nombres, lugares, barcos, acciones) de la consulta original para la búsqueda vectorial en `parsed_text`.

    **Formato de Salida Requerido:** Responde **estrictamente y únicamente** con un objeto JSON válido con las claves "intent" (string: 'text', 'visual' o 'code'), "filters" (dict o null), y "search_query" (string). No incluyas texto fuera del JSON.

    **Ejemplo de Salida JSON:**
    ```json
    {{
    "intent": "text",
    "filters": {{
        "master_name": "Litlejohn"
    }},
    "search_query": "fragata Charles Edwin capitán Litlejohn"
    }}
    Tu Análisis (Solo JSON):
"""

# Crear la plantilla de prompt
prompt_template = ChatPromptTemplate.from_template(MODERATOR_PROMPT)

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analiza la consulta del usuario usando el LLM para determinar
    la intención, extraer filtros de metadatos y generar una consulta
    optimizada para la búsqueda vectorial.

    Args:
        query: La consulta original del usuario en lenguaje natural.

    Returns:
        Un diccionario con las claves:
        - 'intent': 'text', 'visual', o 'code'.
        - 'filters': Un diccionario con los filtros de metadatos o None.
        - 'search_query': La cadena de texto optimizada para búsqueda vectorial.
        En caso de error, devuelve un diccionario de fallback.
    """
    print(f"Moderador: Iniciando análisis para query: '{query}'")

    # Obtener la instancia del LLM (configurado con baja temperatura para precisión)
    llm = get_llm()
    if not llm:
        print("Error Crítico: LLM no disponible para el agente moderador.")
        return {"intent": "text", "filters": None, "search_query": query}

    # Crear la cadena (chain) LangChain uniendo el prompt y el LLM
    chain = prompt_template | llm

    try:
        # Invocar la cadena con la consulta del usuario
        response = chain.invoke({"query": query})
        content = response.content  # Contenido de la respuesta del LLM

        print(f"Moderador: Respuesta cruda del LLM recibida:\n---\n{content}\n---")

        # --- Extracción robusta del JSON ---
        json_str = extract_json(content)
        if json_str is None:
            return {"intent": "text", "filters": None, "search_query": query}

        # Parsear el string JSON extraído
        parsed_response = json.loads(json_str)

        # --- Validación y Limpieza de la Respuesta Parseada ---
        return validate_parsed_response(parsed_response, query)

    except json.JSONDecodeError as e:
        print(f"Error Crítico: Fallo al parsear JSON de la respuesta del LLM: {e}")
        return {"intent": "text", "filters": None, "search_query": query}

    except Exception as e:
        print(f"Error Inesperado en el agente moderador: {e}")
        import traceback
        traceback.print_exc()
        return {"intent": "text", "filters": None, "search_query": query}

def extract_json(content: str) -> Optional[str]:
    """Extrae el bloque JSON de la respuesta del LLM."""
    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    json_match_loose = re.search(r"(\{[\s\S]*?\})", content, re.DOTALL)
    if json_match_loose:
        return json_match_loose.group(1).strip()
    return None

def validate_parsed_response(parsed_response: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Valida y limpia la respuesta parseada."""
    required_keys = ["intent", "filters", "search_query"]
    if not all(k in parsed_response for k in required_keys):
        print("Error: Respuesta JSON del LLM incompleta. Faltan claves.")
        raise ValueError("Respuesta JSON del LLM no tiene la estructura esperada.")

    valid_intents = ["text", "visual", "code"]
    if parsed_response["intent"] not in valid_intents:
        print(f"Advertencia: Intención inválida '{parsed_response['intent']}' recibida del LLM. Usando 'text' por defecto.")
        parsed_response["intent"] = "text"

    filters_value = parsed_response["filters"]
    if filters_value is not None:
        if not isinstance(filters_value, dict):
            print(f"Advertencia: 'filters' no es un diccionario ({type(filters_value)}). Se establecerá a None.")
            parsed_response["filters"] = None
        elif not filters_value:
            parsed_response["filters"] = None

    if not isinstance(parsed_response.get("search_query"), str):
        print(f"Advertencia: 'search_query' no es un string. Usando la query original.")
        parsed_response["search_query"] = query

    print(f"Moderador: Análisis finalizado y validado: {parsed_response}")
    return parsed_response
