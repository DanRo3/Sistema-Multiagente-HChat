# app/core/dataframe_loader.py
import pandas as pd
import os
import logging
from typing import Optional
from app.core.config import settings # Usar la ruta configurada

logger = logging.getLogger(__name__)

_dataframe_instance: Optional[pd.DataFrame] = None

def load_and_preprocess_dataframe() -> Optional[pd.DataFrame]:
    """Carga y preprocesa el DataFrame desde CSV (Singleton)."""
    global _dataframe_instance
    if _dataframe_instance is not None:
        # logger.debug("Devolviendo instancia de DataFrame existente.")
        return _dataframe_instance

    csv_path = settings.CSV_FILE_PATH # Usar la ruta de la configuración
    logger.info(f"Cargando y preprocesando DataFrame desde: {csv_path}")

    if not os.path.exists(csv_path):
         logger.error(f"Error Crítico: El archivo CSV no existe en la ruta: {csv_path}")
         return None
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"DataFrame cargado inicialmente: {len(df)} filas.")

        # --- Preprocesamiento Esencial ---
        logger.info("Realizando preprocesamiento...")
        date_columns = ['publication_date', 'travel_departure_date', 'travel_arrival_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'travel_duration' in df.columns:
             # Extraer solo dígitos, manejar no números resultando en NaN
             df['travel_duration_days'] = pd.to_numeric(df['travel_duration'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
             logger.info("  Columna 'travel_duration_days' (numérica) creada.")

        # Llenar NaNs en columnas clave si es necesario (opcional)
        # cols_to_fill = ['ship_name', 'master_name', 'travel_departure_port']
        # for col in cols_to_fill:
        #      if col in df.columns: df[col] = df[col].fillna('Desconocido')

        _dataframe_instance = df # Almacenar instancia cargada
        logger.info("DataFrame cargado y preprocesado exitosamente.")
        return _dataframe_instance

    except Exception as e:
        logger.exception(f"Error fatal al cargar o preprocesar el DataFrame: {e}")
        _dataframe_instance = None
        return None

def get_dataframe() -> Optional[pd.DataFrame]:
    """Obtiene la instancia cargada y preprocesada del DataFrame."""
    if _dataframe_instance is None:
        load_and_preprocess_dataframe() # Intentar cargar si no lo está
    return _dataframe_instance