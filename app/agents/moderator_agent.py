
import json
import re
from typing import Dict, Optional, Any
from app.core.llm import get_llm# Importa la función para obtener el LLM
from langchain_core.prompts import ChatPromptTemplate # Para crear la plantilla de prompt

# --- Plantilla de Prompt para el Moderador ---
# Esta plantilla guía al LLM sobre cómo analizar la consulta.
# Es CRUCIAL refinarla basada en pruebas con tu LLM y datos.
MODERATOR_PROMPT = """
    **Tarea:** Analiza la consulta del usuario y extrae información estructurada para guiar la búsqueda y respuesta en un sistema de consulta de registros marítimos históricos.

    **Consulta del Usuario:**
    "{query}"

    **Contexto de Datos:** La base de datos contiene registros sobre viajes marítimos con las siguientes columnas de metadatos relevantes para filtrar:
    - `publication_date` (Fecha de publicación, formato YYYY-MM-DD)
    - `news_section` (Sección del periódico)
    # ... (resto de las columnas) ...
    - `master_name` (Nombre del capitán)
    - `parsed_text` (Texto completo del registro, usado para búsqueda vectorial)

    **Instrucciones Detalladas:**
    1.  **Determina la Intención Principal (`intent`):**
        # ... (descripción de intenciones) ...
    2.  **Identifica Filtros de Metadatos (`filters`):**
        *   Busca en la consulta valores específicos que correspondan **exactamente** a alguna de las columnas de metadatos listadas arriba.
        *   Devuelve un diccionario Python donde las claves son los nombres exactos de las columnas de metadatos y los valores son los extraídos de la consulta.
        *   Sé preciso. Ejemplo para "capitán Smith": `{{"master_name": "Smith"}}`. Ejemplo para "salidas desde Nueva York": `{{"travel_departure_port": "Nueva York"}}`. Ejemplo para "barcos tipo fragata": `{{"ship_type": "fragata"}}`. # <-- ASEGÚRATE que estas llaves dobles {{}} estén correctas si necesitas mostrar llaves literales. Si solo son ejemplos, usa simples {}. Revisando de nuevo, creo que aquí deberían ser SIMPLES, ya que son parte de la explicación, no variables. Cambiemos a:
        *   Sé preciso. Ejemplo para "capitán Smith": { "master_name": "Smith" }. Ejemplo para "salidas desde Nueva York": { "travel_departure_port": "Nueva York" }. Ejemplo para "barcos tipo fragata": { "ship_type": "fragata" }.
        *   Si no se identifican filtros claros y específicos, devuelve `null` o un diccionario vacío `{}`. NO inventes filtros.
    3.  **Crea una Consulta de Búsqueda Vectorial (`search_query`):**
        # ... (descripción de search_query) ...
        *   Ejemplo: Si la consulta es "Información sobre el cargamento de la fragata Charles Edwin capitaneada por Litlejohn saliendo de Nueva York", una buena `search_query` podría ser `"fragata Charles Edwin Litlejohn carga Nueva York"`.

    **Formato de Salida Requerido:** Responde **estrictamente y únicamente** con un objeto JSON válido que contenga las claves "intent" (string: 'text', 'visual' o 'code'), "filters" (dict o null), y "search_query" (string). No incluyas explicaciones adicionales fuera del JSON.

    **Ejemplo de Salida JSON:**
    ```json
    {
    "intent": "text",
    "filters": {
        "master_name": "Litlejohn",
        "ship_type": "frat, am."
    },
    "search_query": "fragata americana Charles Edwin capitán Litlejohn"
    }
    ```

    Tu Análisis (Solo JSON):
"""


prompt_template = ChatPromptTemplate.from_template(MODERATOR_PROMPT)

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analiza la consulta del usuario usando el LLM (Gemini) para determinar
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
        # Fallback MUY básico si el LLM falla completamente
        return {"intent": "text", "filters": None, "search_query": query}

    # Crear la cadena (chain) LangChain uniendo el prompt y el LLM
    chain = prompt_template | llm

    try:
        # Invocar la cadena con la consulta del usuario
        response = chain.invoke({"query": query})
        content = response.content # Contenido de la respuesta del LLM

        print(f"Moderador: Respuesta cruda del LLM recibida:\n---\n{content}\n---")

        # --- Extracción robusta del JSON ---
        # 1. Buscar bloque JSON delimitado por ```json ... ```
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            print(f"Moderador: Bloque JSON extraído:\n---\n{json_str}\n---")
        else:
            # 2. Si no hay bloque, intentar encontrar un JSON que empiece con { y termine con }
            #    Esto es menos preciso pero puede funcionar si el LLM solo devuelve JSON.
            json_match_loose = re.search(r"(\{[\s\S]*?\})", content, re.DOTALL)
            if json_match_loose:
                json_str = json_match_loose.group(1).strip()
                print(f"Moderador: JSON encontrado sin delimitadores:\n---\n{json_str}\n---")
            else:
                # 3. Si todo falla, asumir que la respuesta podría ser JSON directamente (último recurso)
                json_str = content.strip()
                print(f"Moderador: No se encontraron delimitadores JSON claros, intentando parsear toda la respuesta.")
        # ------------------------------------

        # Parsear el string JSON extraído
        parsed_response = json.loads(json_str)

        # --- Validación y Limpieza de la Respuesta Parseada ---
        # Verificar claves esenciales
        required_keys = ["intent", "filters", "search_query"]
        if not all(k in parsed_response for k in required_keys):
            print("Error: Respuesta JSON del LLM incompleta. Faltan claves.")
            raise ValueError("Respuesta JSON del LLM no tiene la estructura esperada.")

        # Validar valor de 'intent'
        valid_intents = ["text", "visual", "code"]
        if parsed_response["intent"] not in valid_intents:
            print(f"Advertencia: Intención inválida '{parsed_response['intent']}' recibida del LLM. Usando 'text' por defecto.")
            parsed_response["intent"] = "text"

        # Validar y limpiar 'filters'
        filters_value = parsed_response["filters"]
        if filters_value is not None:
            if not isinstance(filters_value, dict):
                print(f"Advertencia: 'filters' no es un diccionario ({type(filters_value)}). Se establecerá a None.")
                parsed_response["filters"] = None
            elif not filters_value: # Si es un diccionario vacío {}
                parsed_response["filters"] = None # Estandarizar a None
        # Si es None o un diccionario válido (no vacío), se mantiene.

        # Asegurar que search_query sea string
        if not isinstance(parsed_response.get("search_query"), str):
            print(f"Advertencia: 'search_query' no es un string. Usando la query original.")
            parsed_response["search_query"] = query # Fallback

        # ----------------------------------------------------

        print(f"Moderador: Análisis finalizado y validado: {parsed_response}")
        return parsed_response

    except json.JSONDecodeError as e:
        print(f"Error Crítico: Fallo al parsear JSON de la respuesta del LLM: {e}")
        print(f"Respuesta LLM que causó el error:\n{content}")
        # Fallback: usa la query original, sin filtros, intención texto
        return {"intent": "text", "filters": None, "search_query": query}

    except Exception as e:
        # Captura cualquier otro error inesperado durante la lógica del agente
        print(f"Error Inesperado en el agente moderador: {e}")
        import traceback
        traceback.print_exc() # Imprime el stack trace para depuración
        return {"intent": "text", "filters": None, "search_query": query}

