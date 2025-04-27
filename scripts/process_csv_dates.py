import pandas as pd

# Cargar el CSV original
df = pd.read_csv('data/DataLimpia.csv')

# Convertir las columnas de fechas a datetime, coercitendo errores a NaT
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df['travel_departure_date'] = pd.to_datetime(df['travel_departure_date'], errors='coerce')
df['travel_arrival_date'] = pd.to_datetime(df['travel_arrival_date'], errors='coerce')

# Guardar el nuevo CSV procesado
df.to_csv('data/DataLimpia_procesada.csv', index=False)