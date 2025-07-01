# app/pandasai_utils/skills.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Optional, Dict, Any # Asegúrate de importar List, Dict, Any
from pandasai.skills import skill
from app.core.config import settings # Para PANDASAI_CHART_DIR_NAME

logger = logging.getLogger(__name__)

# --- Mapeo de Tipos de Barco (Centralizado) ---
# Considera mover esto a un archivo de configuración o un módulo de constantes si es grande
MAPEO_TIPOS_BARCO: Dict[str, str] = {
    "berg. am.": "Bergantín Americano",
    "frag. esp.": "Fragata Española",
    "vapor am.": "Vapor Americano",
    "berg. esp.": "Bergantín Español",
    "frag. am.": "Fragata Americana",
    "berg. ing.": "Bergantín Inglés",
    "pol. esp.": "Polacra Española",
    "gol. am.": "Goleta Americana",
    "corb. am.": "Corbeta Americana",
    "berg. goe. am.": "Bergantín Goleta Americano",
    # ... Añade todos tus mapeos aquí ...
    "nan": "No especificado" # Ejemplo para manejar NaN si aparece
}
logger.info(f"Cargados {len(MAPEO_TIPOS_BARCO)} mapeos de tipos de barco para skills.")

# --- Skill para Obtener Datos Tabulares ---
@skill
def get_tabular_data(
    df: pd.DataFrame,
    columns_to_select: Optional[List[str]] = None,
    filter_conditions: Optional[str] = None,
    sort_by: Optional[List[Dict[str, str]]] = None,
    limit: Optional[int] = None,
    query_description: Optional[str] = "Obtener datos tabulares"
) -> pd.DataFrame:
    """
        HABILIDAD PERSONALIZADA: Recupera datos tabulares filtrados y seleccionados de un DataFrame.
        Esta habilidad es preferible para consultas que requieren devolver un subconjunto de datos
        en formato de DataFrame completo.
        Parámetros:
        - df: El DataFrame de entrada (PandasAI lo pasará automáticamente).
        - columns_to_select: Lista de columnas a devolver.
        - filter_conditions: String de query de Pandas para filtrar.
        - sort_by: Lista de dicts para ordenar.
        - limit: Número de filas.
        - query_description: Descripción.
        Devuelve: Un DataFrame de Pandas con los resultados.
    """
    df_result = df.copy()
    logger.info(f"[Skill:get_tabular_data] Iniciando: {query_description}")

    # 1. Aplicar Filtros
    if filter_conditions and filter_conditions.strip():
        try:
            logger.debug(f"  Aplicando filtro: {filter_conditions}")
            df_result = df_result.query(filter_conditions)
            logger.info(f"  Filas después del filtro: {len(df_result)}")
            if df_result.empty:
                logger.warning("  DataFrame vacío después del filtro. No se realizarán más operaciones.")
                return pd.DataFrame(columns=df.columns if columns_to_select is None else columns_to_select) # Devuelve con columnas esperadas
        except Exception as e:
            logger.error(f"  Error aplicando filtro '{filter_conditions}': {e}. Devolviendo DataFrame vacío.")
            return pd.DataFrame(columns=df.columns if columns_to_select is None else columns_to_select)

    # 2. Seleccionar Columnas (después de filtrar, sobre el resultado)
    if columns_to_select:
        valid_columns = [col for col in columns_to_select if col in df_result.columns]
        if not valid_columns:
            logger.warning(f"  Ninguna de las columnas solicitadas para seleccionar ({columns_to_select}) existe en el DataFrame filtrado. Se devolverán todas las columnas disponibles del filtro o un DF vacío.")
            # Si no hay columnas válidas, pero el DF filtrado tiene columnas, devolver esas.
            # Si el DF filtrado está vacío, df_result.columns estará vacío.
        elif len(valid_columns) < len(columns_to_select):
            missing = set(columns_to_select) - set(valid_columns)
            logger.warning(f"  Algunas columnas solicitadas no se encontraron/fueron inválidas: {missing}. Seleccionando: {valid_columns}")
            df_result = df_result[valid_columns]
        else:
            logger.debug(f"  Seleccionando columnas: {valid_columns}")
            df_result = df_result[valid_columns]
    
    # 3. Ordenar Datos
    if sort_by and not df_result.empty:
        sort_columns = []
        sort_orders_bool = []
        for Sorter in sort_by:
            col = Sorter.get("column")
            order = Sorter.get("order", "asc").lower()
            if col in df_result.columns:
                sort_columns.append(col)
                sort_orders_bool.append(order == "asc")
            else:
                logger.warning(f"  Columna para ordenar '{col}' no encontrada. Se ignora.")
        
        if sort_columns:
            try:
                logger.debug(f"  Ordenando por: {sort_columns}, Órdenes: {sort_orders_bool}")
                df_result = df_result.sort_values(by=sort_columns, ascending=sort_orders_bool)
            except Exception as e:
                logger.error(f"  Error al ordenar: {e}. Se continúa sin ordenar.")

    # 4. Limitar Resultados
    if limit is not None and limit > 0 and not df_result.empty:
        logger.debug(f"  Limitando a {limit} filas.")
        df_result = df_result.head(limit)

    logger.info(f"[Skill:get_tabular_data] Finalizado. Devolviendo DataFrame con {len(df_result)} filas y {len(df_result.columns)} columnas.")
    return df_result


# --- Skill para Generar Gráficos de Frecuencia (Top N) ---
@skill
def plot_top_n_frequencies(
    df: pd.DataFrame,
    column_name: str,
    top_n: int = 15,
    chart_title: Optional[str] = None,
    normalize_ship_types: bool = False, # Específico para ship_type
    query_description: Optional[str] = "Graficar frecuencias Top N"
) -> Optional[str]: # Devuelve la ruta al archivo del gráfico o un mensaje de error/None
    """
    Genera un gráfico de barras de las 'top_n' frecuencias más comunes para 'column_name'.
    Guarda el gráfico y devuelve la ruta.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        column_name (str): Nombre de la columna para calcular frecuencias.
        top_n (int): Número de los elementos más frecuentes a mostrar.
        chart_title (Optional[str]): Título personalizado para el gráfico.
        normalize_ship_types (bool): Si es True y column_name es 'ship_type', normaliza los nombres.
        query_description (Optional[str]): Descripción para logging.

    Returns:
        Optional[str]: Ruta al archivo del gráfico guardado, o un string de error/None.
    """
    logger.info(f"[Skill:plot_top_n_frequencies] Iniciando: {query_description} para columna '{column_name}' (Top {top_n})")
    try:
        if column_name not in df.columns:
            msg = f"Columna '{column_name}' no encontrada en el DataFrame."
            logger.error(f"  {msg}")
            return msg # Devolver mensaje de error

        if df[column_name].isnull().all():
            msg = f"La columna '{column_name}' solo contiene valores nulos. No se puede generar el gráfico."
            logger.warning(f"  {msg}")
            return msg

        counts = df[column_name].value_counts(dropna=True).nlargest(top_n)

        if counts.empty:
            msg = f"No hay datos para graficar para la columna '{column_name}' después de contar (Top {top_n})."
            logger.warning(f"  {msg}")
            return msg
        
        logger.debug(f"  Frecuencias calculadas (Top {top_n}):\n{counts.head()}")

        # Normalizar nombres si es ship_type y se solicita
        if normalize_ship_types and column_name == 'ship_type':
            logger.info("  Normalizando nombres de ship_type para el gráfico.")
            original_indices = counts.index.tolist()
            counts.index = counts.index.map(lambda x: MAPEO_TIPOS_BARCO.get(str(x).strip(), str(x).strip()))
            logger.debug(f"    Índices originales: {original_indices}, Índices normalizados: {counts.index.tolist()}")


        final_chart_title = chart_title or f'Top {top_n} Frecuencias de {column_name.replace("_", " ").title()}'
        
        # Mejoras visuales para el gráfico
        plt.figure(figsize=(max(10, int(len(counts)*0.5) ), 6)) # Ancho dinámico, mínimo 10
        bars = counts.plot(kind='bar', color='skyblue', width=0.85)
        plt.title(final_chart_title, fontsize=15, pad=20)
        plt.ylabel('Frecuencia', fontsize=11)
        plt.xlabel(column_name.replace("_", " ").title(), fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Añadir valores encima de las barras
        for bar in bars.patches:
            bars.annotate(format(bar.get_height(), '.0f'),
                           (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                           size=8, xytext=(0, 8),
                           textcoords='offset points')
            
        plt.tight_layout(pad=1.5)

        chart_filename = f"plot_top_n_{column_name.replace(' ', '_').replace('.', '')}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}.png"
        chart_save_path = os.path.join(settings.PANDASAI_CHART_DIR_NAME, chart_filename)
        
        os.makedirs(settings.PANDASAI_CHART_DIR_NAME, exist_ok=True)
        plt.savefig(chart_save_path, dpi=100) # Guardar con buena resolución
        plt.close()
        logger.info(f"  Gráfico guardado en: {chart_save_path}")
        
        return chart_save_path

    except Exception as e:
        logger.exception(f"[Skill:plot_top_n_frequencies] Error crítico generando gráfico para '{column_name}': {e}")
        return f"Error al generar el gráfico para '{column_name}': {str(e)[:100]}" # Devolver mensaje de error