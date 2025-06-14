# Propongo crear un nuevo archivo, ej: app/pandasai_utils/response_parsers.py
import pandas as pd
from pandasai.responses.response_parser import ResponseParser
import logging

logger = logging.getLogger(__name__)

class FullDataFrameResponseParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
        logger.debug("FullDataFrameResponseParser inicializado.")

    def format_dataframe(self, result: dict) -> list[dict] | pd.DataFrame:
        """
        Asegura que los DataFrames se devuelvan como una lista completa de diccionarios.
        PandasAI v2 pasa un dict con 'type': 'dataframe' y 'value': pd.DataFrame.
        """
        try:
            df_value = result.get("value")
            if isinstance(df_value, pd.DataFrame):
                logger.info(f"FullDataFrameResponseParser: Formateando DataFrame de {len(df_value)} filas a lista de diccionarios.")
                return df_value.to_dict(orient='records')
            elif isinstance(df_value, pd.Series):
                logger.info(f"FullDataFrameResponseParser: Formateando Series de {len(df_value)} elementos a lista de diccionarios.")
                # Convertir Series a DataFrame antes para un formato consistente
                return df_value.to_frame(name=df_value.name or 'value').to_dict(orient='records')
            else:
                logger.warning(f"FullDataFrameResponseParser: Se esperaba pd.DataFrame o pd.Series en 'value', se obtuvo {type(df_value)}. Devolviendo como está.")
                return df_value # Devolver el valor original si no es DataFrame/Series
        except Exception as e:
            logger.exception(f"FullDataFrameResponseParser: Error formateando DataFrame: {e}")
            return result.get("value") # Intentar devolver el valor original en caso de error

    def format_string(self, result: dict) -> str:
        """Devuelve el string tal cual, evitando resúmenes no deseados si PandasAI ya lo hizo."""
        value = result.get("value", "")
        logger.info(f"FullDataFrameResponseParser: Formateando string. Longitud: {len(str(value))}")
        return str(value)

    def format_number(self, result: dict) -> (int | float):
        """Devuelve el número tal cual."""
        value = result.get("value")
        logger.info(f"FullDataFrameResponseParser: Formateando número: {value}")
        return value

    # format_plot puede ser útil si quieres cambiar cómo se maneja la ruta del plot,
    # pero tu lógica actual ya lo maneja bien.