# app/utils/json_parser.py
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_json(content: str) -> Optional[str]:
    """
    Extrae el primer bloque JSON ```json ... ``` o el primer objeto JSON { ... }
    de un string de respuesta de LLM.
    """
    if not content:
        return None

    # Priorizar bloque delimitado con ```json
    json_block_match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", content, re.DOTALL)
    if json_block_match:
        extracted = json_block_match.group(1).strip()
        logger.debug(f"JSON extraído (patrón ```json): {extracted[:100]}...")
        return extracted

    # Si no, buscar si la respuesta es SOLO un objeto JSON
    stripped_content = content.strip()
    if stripped_content.startswith('{') and stripped_content.endswith('}'):
        # Verificar si es un JSON válido antes de devolverlo
        try:
            import json
            json.loads(stripped_content) # Intenta parsear para validar
            logger.debug("Respuesta parece ser solo un objeto JSON válido.")
            return stripped_content
        except json.JSONDecodeError:
             logger.warning("La cadena parece un objeto JSON pero no se pudo parsear.")
             # Intentar encontrar el primer { y el último } si falla el parseo directo
             first_brace = stripped_content.find('{')
             last_brace = stripped_content.rfind('}')
             if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                  potential_json = stripped_content[first_brace:last_brace+1]
                  try:
                       json.loads(potential_json)
                       logger.debug("JSON extraído encontrando primer/último '{' y '}'.")
                       return potential_json
                  except json.JSONDecodeError:
                       logger.warning("Fallo al extraer JSON incluso con primer/último '{' y '}'.")


    logger.warning("No se encontró un bloque JSON reconocible en la respuesta.")
    return None # No encontrado o no válido