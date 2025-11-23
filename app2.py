import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("🚗 Predicción del Precio de Carros (Random Forest)")
st.subheader("Utilice la barra lateral para ingresar las características del vehículo.")

# ----------------------------------------------------------------------
# 1. FUNCIÓN DE CARGA Y ENTRENAMIENTO CACHEADO
# Se usa st.cache_resource para entrenar solo una vez en el servidor
# ----------------------------------------------------------------------

@st.cache_resource 
def load_and_train_model():
    """Carga los datos, aplica el preprocesamiento completo, entrena el modelo 
    y retorna el modelo ajustado, las columnas de entrenamiento y el dataframe base."""
    st.info("Iniciando carga y entrenamiento del modelo (Esto solo ocurre la primera vez).")
    
    try:
        # Cargar el dataset (debe estar en la misma carpeta del repositorio)
        df = pd.read_csv('car_price_prediction.csv') # [2]
    except FileNotFoundError:
        st.error("Error: Archivo 'car_price_prediction.csv' no encontrado.")
        return None, None, None

    # --- PREPROCESAMIENTO Y FEATURE ENGINEERING ---
    
    # Limpieza de 'Levy'
    df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce') # [2]
    df['Levy'].fillna(0, inplace=True) # [4]
    
    # Ingeniería de característica 'Is_Turbo' y limpieza de 'Engine volume'
    df['Is_Turbo'] = np.where(df['Engine volume'].astype(str).str.contains('Turbo', na=False), 1, 0) # [4]
    df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '', regex=False) # [4]
    df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce') # [4]
    
    # Limpieza de 'Mileage'
    df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False) # [4]
    df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce') # [5]

    # --- FILTRADO DE OUTLIERS ---
    
    # 1. Filtrar Price >= 100 [5]
    df = df[df['Price'] >= 100].copy() 

    # 2. Filtrado de Outliers para 'Price' (usando IQR)
    Q1_P = df['Price'].quantile(0.25) # [5]
    Q3_P = df['Price'].quantile(0.75) # [5]
    IQR_P = Q3_P - Q1_P
    lower_bound_iqr_P = Q1_P - 1.5 * IQR_P # [6]
    upper_bound_iqr_P = Q3_P + 1.5 * IQR_P # [6]
    df_cleaned = df[(df['Price'] >= lower_bound_iqr_P) & (df['Price'] <= upper_bound_iqr_P)].copy() # [6]

    # 3. Filtrado de Outliers para 'Levy' (usando IQR)
    Q1_L = df_cleaned['Levy'].quantile(0.25) # [6]
    Q3_L = df_cleaned['Levy'].quantile(0.75) # [6]
    IQR_L = Q3_L - Q1_L
    lower_bound_iqr_L = Q1_L - 1.5 * IQR_L # [7]
    upper_bound_iqr_L = Q3_L + 1.5 * IQR_L # [7]
    df_cleaned = df_cleaned[(df_cleaned['Levy'] >= lower_bound_iqr_L) & (df_cleaned['Levy'] <= upper_bound_iqr_L)].copy() # [7]

    # --- SELECCIÓN Y CODIFICACIÓN ---
    
    selected_columns = ['Price', 'Manufacturer', 'Model', 'Prod. year', 'Category', 'Fuel type', 'Gear box type']
    df_selected = df_cleaned[selected_columns].copy() # [7]

    y = df_selected['Price'] # [7]
    
    numerical_cols = ['Prod. year'] # Columnas numéricas usadas en la matriz X [8]
    categorical_cols = df_selected.select_dtypes(include=['object']).columns.tolist() # [8]
    
    # Codificación One-Hot [3]
    df_categorical_encoded = pd.get_dummies(df_selected[categorical_cols], drop_first=True) 
    
    # Construcción de la matriz X
    X = pd.concat([df_selected[numerical_cols], df_categorical_encoded], axis=1) # [3]
    training_columns = X.columns.tolist()

    # --- ENTRENAMIENTO DEL MODELO ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # [3]
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # [3]
    model_rf.fit(X_train, y_train) # [9]
    
    st.success("Modelo entrenado y recursos listos.")
    
    # Retornamos el modelo, las columnas de entrenamiento y el DataFrame seleccionado (para widgets)
    return model_rf, training_columns, df_selected

# Cargar/Entrenar el modelo
model_rf, training_columns, df_selected = load_and_train_model()

# -----------------------------
# 2. INTERFAZ DE ENTRADA DE USUARIO (WIDGETS) [10]
# -----------------------------

if model_rf is not None and not df_selected.empty:
    
    st.sidebar.header("Variables de Predicción") # [10]

    # Inputs Numéricos
    min_year = int(df_selected['Prod. year'].min())
    max_year = int(df_selected['Prod. year'].max())

    prod_year = st.sidebar.number_input(
        "Año de Producción (Prod. year):", 
        min_value=min_year,
        max_value=max_year,
        value=2015,
        step=1
    )
    
    # Inputs Categóricos
    
    manufacturer = st.sidebar.selectbox(
        "Fabricante (Manufacturer):",
        options=df_selected["Manufacturer"].unique()
    )
    
    model_car = st.sidebar.selectbox(
        "Modelo (Model):",
        options=df_selected["Model"].unique()
    )
    
    category = st.sidebar.selectbox(
        "Categoría:",
        options=df_selected["Category"].unique()
    )
    
    fuel_type = st.sidebar.selectbox(
        "Tipo de Combustible (Fuel type):",
        options=df_selected["Fuel type"].unique()
    )
    
    gear_box_type = st.sidebar.selectbox(
        "Tipo de Caja de Cambios (Gear box type):",
        options=df_selected["Gear box type"].unique()
    )

    # Input de Feature Engineered (Is_Turbo)
    is_turbo = st.sidebar.checkbox("¿Es Turbo?", value=False)
    
    if st.button("Obtener Predicción"):
        
        # -----------------------------
        # 3. CONSTRUCCIÓN Y PREDICCIÓN
        # -----------------------------
        
        data_input = {
            'Prod. year': [prod_year],
            'Manufacturer': [manufacturer],
            'Model': [model_car],
            'Category': [category],
            'Fuel type': [fuel_type],
            'Gear box type': [gear_box_type]
        }
        input_df = pd.DataFrame(data_input)
        
        # Aplicar la Codificación One-Hot
        categorical_cols_app = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type']
        input_dummies = pd.get_dummies(input_df[categorical_cols_app], drop_first=True)
        
        # Preparar el DataFrame final X_pred usando las training_columns
        X_pred = pd.DataFrame(data=0, index=[0], columns=training_columns) 
        
        # Mapear valores numéricos
        X_pred['Prod. year'] = prod_year
        
        # Mapear característica Is_Turbo
        if 'Is_Turbo' in training_columns: 
             X_pred['Is_Turbo'] = 1 if is_turbo else 0
             
        # Mapear variables categóricas (One-Hot)
        for col in input_dummies.columns:
            if col in training_columns:
                # Usa iloc para obtener el valor del input_dummies
                X_pred[col] = input_dummies[col].iloc

        # Realizar la Predicción
        try:
            # Predicción similar a la lógica vista en la fuente [11]
            prediction = model_rf.predict(X_pred)[0]
            
            # Mostrar el resultado
            st.subheader("Resultado de la Predicción")
            st.success(f"**Predicción del precio del vehículo (USD): ${prediction:,.2f}**")
            
        except ValueError as e:
            st.error(f"Error al realizar la predicción. Detalle: {e}")

else:
    st.warning("No se pudo iniciar la aplicación. Revise si el archivo 'car_price_prediction.csv' está en el directorio.")
