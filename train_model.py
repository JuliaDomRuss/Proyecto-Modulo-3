#%%writefile train_model.py
import numpy as np
import pandas as pd
import joblib # Necesario para guardar el modelo de scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# Importaciones de métricas no necesarias para el entrenamiento, pero incluidas por completitud
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

print("Iniciando proceso de entrenamiento...")

# 1. CARGA DE DATOS Y PREPROCESAMIENTO
# --------------------------------------------------------------------------------

# Cargar el dataset
try:
    df = pd.read_csv('car_price_prediction.csv') # [1]
except FileNotFoundError:
    print("Error: 'car_price_prediction.csv' no encontrado.")
    exit()

# Conversión y limpieza de 'Levy'
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce') # [1]
df['Levy'].fillna(0, inplace=True) # [2]

# Ingeniería de característica 'Is_Turbo' y limpieza de 'Engine volume'
df['Is_Turbo'] = np.where(df['Engine volume'].astype(str).str.contains('Turbo', na=False), 1, 0) # [2]
df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '', regex=False) # [2]
df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce') # [2]

# Limpieza y conversión de 'Mileage'
df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False) # [2]
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce') # [3]

# 2. FILTRADO DE OUTLIERS
# --------------------------------------------------------------------------------

# Filtrar df para remover precios menores a 100 [3]
df = df[df['Price'] >= 100].copy() 

# Filtrado de Outliers para 'Price' (usando IQR)
Q1_P = df['Price'].quantile(0.25) # [3]
Q3_P = df['Price'].quantile(0.75) # [3]
IQR_P = Q3_P - Q1_P
lower_bound_iqr_P = Q1_P - 1.5 * IQR_P # [4]
upper_bound_iqr_P = Q3_P + 1.5 * IQR_P # [4]
df_cleaned = df[(df['Price'] >= lower_bound_iqr_P) & (df['Price'] <= upper_bound_iqr_P)].copy() # [4]

# Filtrado de Outliers para 'Levy' (usando IQR)
Q1_L = df_cleaned['Levy'].quantile(0.25) # [4]
Q3_L = df_cleaned['Levy'].quantile(0.75) # [4]
IQR_L = Q3_L - Q1_L
lower_bound_iqr_L = Q1_L - 1.5 * IQR_L # [4]
upper_bound_iqr_L = Q3_L + 1.5 * IQR_L # [5]
df_cleaned = df_cleaned[(df_cleaned['Levy'] >= lower_bound_iqr_L) & (df_cleaned['Levy'] <= upper_bound_iqr_L)].copy() # [5]


# 3. SELECCIÓN DE CARACTERÍSTICAS Y CODIFICACIÓN
# --------------------------------------------------------------------------------

# Selección de columnas clave para el modelo
selected_columns = ['Price', 'Manufacturer', 'Model', 'Prod. year', 'Category', 'Fuel type', 'Gear box type']
df_selected = df_cleaned[selected_columns].copy() # [5]

# Definición de variables X e y
y = df_selected['Price'] # [5]

# Identificar columnas numéricas y categóricas según las seleccionadas [6]
numerical_cols = ['Prod. year'] # El source indica que estas columnas son usadas para la codificación [7]
categorical_cols = df_selected.select_dtypes(include=['object']).columns.tolist() # [6]

# Codificación One-Hot para variables categóricas
df_categorical_encoded = pd.get_dummies(df_selected[categorical_cols], drop_first=True) # [6]

# Construcción de la matriz de características X
X = pd.concat([df_selected[numerical_cols], df_categorical_encoded], axis=1) # [7]

# Guardar la lista de columnas, crucial para la predicción en la app
training_columns = X.columns.tolist()


# 4. ENTRENAMIENTO DEL MODELO
# --------------------------------------------------------------------------------

# División de datos (aunque no se necesita X_test/y_test para la app, se requiere para el entrenamiento según el source)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # [7]

# Inicializar y entrenar el modelo RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # [7]
model_rf.fit(X_train, y_train) # [8]

print("Modelo RandomForestRegressor entrenado exitosamente.")

# 5. GUARDAR EL MODELO Y LAS COLUMNAS
# --------------------------------------------------------------------------------

# Guardar el modelo ajustado
joblib.dump(model_rf, 'model_rf.joblib')
print("Modelo guardado como 'model_rf.joblib'")

# Guardar la lista de columnas de entrenamiento
joblib.dump(training_columns, 'training_columns.joblib')
print("Lista de columnas de entrenamiento guardada como 'training_columns.joblib'")
