import json
import re
import logging
from typing import Dict, Optional, Any
from app.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# --- Prompt Mejorado para Enrutamiento ---

SYSTEM_INSTRUCTIONS = """**Tarea:** Eres un asistente experto en traducir consultas de usuarios sobre registros históricos marítimos a una `pandasai_query` clara y una `intent`. La `pandasai_query` será ejecutada por PandasAI. **El DataFrame estará disponible en el código generado por PandasAI bajo el nombre `df`**.

**Contexto de Datos:** El DataFrame (`df`) contiene registros con columnas: `publication_date` (datetime), `news_section`, `travel_departure_date` (datetime), `travel_duration` (texto), `travel_arrival_date` (datetime), `travel_departure_port`, `travel_port_of_call_list`, `travel_arrival_port`, `ship_type`, `ship_name`, `cargo_list`, `master_role`, `master_name`, `parsed_text`, `travel_duration_days` (float).

**Instrucciones Detalladas:**
1.  **Determina la Intención Final (`intent`):** Clasifica la *solicitud original del usuario* como 'visual' si pide explícitamente un gráfico/tabla/plot, o 'text' en los demás casos.
2.  **Formula la Consulta para PandasAI (`pandasai_query`):**
    *   Reformula la consulta original del usuario en una instrucción clara y directa para PandasAI, **asegurándote de que las operaciones de Pandas se realicen sobre la variable `df`**.
    *   Si el usuario pide datos específicos: "Filtra df donde [...] y devuelve un DataFrame con [...]".
    *   Si pide un cálculo: "Calcula el promedio de df['columna'] donde [...] y devuelve solo el número."
    *   Si pide un gráfico: "Genera un gráfico de barras de df[...] y guárdalo."
    *   Si la consulta es sobre `parsed_text`: "Devuelve un DataFrame con columnas [...] de df donde df['parsed_text'] contenga 'palabra'."

**Formato de Salida Requerido:** Responde **únicamente** con un objeto JSON válido con las claves: "intent" (string: 'text' o 'visual') y "pandasai_query" (string).


**Ejemplo Salida 1 (Pide Datos):**
```json
{{
  "intent": "text",
  "pandasai_query": "Filtra df para encontrar registros donde df['master_name'] sea 'Smith' y df['travel_departure_port'] sea 'Nueva York', luego devuelve un DataFrame con las columnas 'ship_name' y 'publication_date' de esos registros."
}}
```

**Ejemplo Salida 2 (Pide Conteo):**

```
{{
  "intent": "text",
  "pandasai_query": "Cuenta el número de barcos únicos (ship_name) que llegaron al puerto de 'La Habana' en 1855 y devuelve solo el número."
}}
```

**Ejemplo Salida 3 (Pide Gráfico):**

```
{{
  "intent": "visual",
  "pandasai_query": "Genera un gráfico de barras mostrando la frecuencia de los 5 puertos de salida (travel_departure_port) más comunes, intenta usar diferentes colores en las graficas y guárdalo."
}}
```
**Ejemplo Salida 4 (Busca en Texto y pide columnas específicas):**

{{
  "intent": "text",
  "pandasai_query": "Devuelve un DataFrame con las columnas 'ship_name' y 'parsed_text' de df donde la columna df['parsed_text'] contenga la palabra 'tormenta'."
}}
"""

HUMAN_TASK_PREFIX = """**Consulta del Usuario:**
"{query}"

**Tu Análisis (Solo JSON):**
"""

# Crear plantilla (usando from_messages para robustez)
prompt_template = None
try:
    base_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", HUMAN_TASK_PREFIX)
    ])
    base_template.input_variables = ["query"]
    prompt_template = base_template
    logger.info("Plantilla de prompt del Moderador creada.")
except Exception as e:
    logger.exception(f"Error CRÍTICO al crear ChatPromptTemplate para Moderador: {e}.")

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analiza la consulta, determina la intención final, y genera
    la consulta optimizada para PandasAI.
    """
    logger.info(f"Moderador: Iniciando análisis para query: '{query}'")

    if prompt_template is None:
         logger.error("Plantilla de prompt del Moderador inválida. Usando fallback.")
         return {"intent": "text", "pandasai_query": query}

    llm = get_llm()
    if not llm:
        logger.error("Error Crítico: LLM no disponible para el agente moderador.")
        return {"intent": "text", "pandasai_query": query}

    chain = prompt_template | llm

    try:
        logger.debug("Invocando cadena del moderador...")
        response = chain.invoke({"query": query})
        content = response.content
        logger.info(f"Moderador: Respuesta cruda del LLM (primeros 500 chars):\n---\n{content[:500]}\n---")

        json_str = extract_json(content)
        if json_str is None:
            logger.error("No se pudo extraer JSON de la respuesta del LLM moderador.")
            return {"intent": "text", "pandasai_query": query}

        logger.debug(f"Moderador: String JSON a parsear:\n---\n{repr(json_str)}\n---")
        parsed_response = json.loads(json_str)
        validated_response = validate_parsed_pandasai_response(parsed_response, query)
        logger.info(f"Moderador: Análisis finalizado: {validated_response}")
        return validated_response

    except json.JSONDecodeError as e:
        logger.error(f"Error Crítico: Fallo al parsear JSON: {e}")
        logger.error(f"Respuesta LLM que causó el error:\n{content}")
        return {"intent": "text", "pandasai_query": query}
    except ValueError as e:
        logger.error(f"Error durante la validación de la respuesta parseada: {e}")
        return {"intent": "text", "pandasai_query": query}
    except Exception as e:
        logger.exception(f"Error Inesperado en el agente moderador: {e}")
        return {"intent": "text", "pandasai_query": query}

def extract_json(content: str) -> Optional[str]:
    """Extrae el primer bloque JSON ```json ... ``` o el primer objeto JSON { ... }."""
    json_block_match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", content, re.DOTALL)
    if json_block_match:
        extracted = json_block_match.group(1).strip()
        logger.debug(f"JSON extraído (```json): {extracted[:100]}...")
        return extracted
    stripped_content = content.strip()
    # Buscar un JSON que ocupe toda la línea o esté indentado
    json_object_match = re.search(r"^\s*(\{[\s\S]*?\})\s*$", stripped_content, re.MULTILINE)
    if json_object_match:
        logger.debug("Respuesta parece ser solo un objeto JSON.")
        return json_object_match.group(1).strip()
    logger.warning("No se encontró un bloque JSON reconocible en la respuesta.")
    return None

def validate_parsed_pandasai_response(parsed_response: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    """Valida la respuesta JSON parseada del LLM moderador (versión PandasAI)."""
    required_keys = ["intent", "pandasai_query"]
    if not all(k in parsed_response for k in required_keys):
        missing_keys = [k for k in required_keys if k not in parsed_response]
        logger.error(f"Respuesta JSON incompleta. Faltan: {missing_keys}. Recibido: {parsed_response.keys()}")
        raise ValueError("Respuesta JSON del moderador incompleta.")

    valid_intents = ["text", "visual", "code"] # Mantener 'code' por si acaso
    intent = parsed_response.get("intent")
    if intent not in valid_intents:
        logger.warning(f"Intención inválida '{intent}'. Usando 'text'.")
        parsed_response["intent"] = "text"

    pandasai_query = parsed_response.get("pandasai_query")
    if not isinstance(pandasai_query, str) or not pandasai_query.strip():
        logger.warning(f"'pandasai_query' inválido o vacío. Usando original.")
        parsed_response["pandasai_query"] = original_query

    # Añadir claves faltantes con None para consistencia del estado
    parsed_response.setdefault("filters", None) # Ya no lo generamos pero lo mantenemos None
    parsed_response.setdefault("search_query", None) # Ya no lo generamos

    return parsed_response