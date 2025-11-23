#%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib 

st.title("🚗 Predicción del Precio de Carros (Random Forest)")
st.subheader("Utilice la barra lateral para ingresar las características del vehículo.")

# -----------------------------
# 1. CARGA DE RECURSOS DEL MODELO Y DATOS PARA WIDGETS
# -----------------------------

@st.cache_resource 
def load_model_resources():
    """Carga el modelo ajustado y la lista de columnas de entrenamiento."""
    try:
        # Cargar el modelo ajustado (RandomForestRegressor)
        model_rf = joblib.load('model_rf.joblib')
        # Cargar la lista de columnas (Crucial para el One-Hot Encoding)
        training_columns = joblib.load('training_columns.joblib')
        st.success("Modelo y estructura cargados correctamente.")
        return model_rf, training_columns
    except FileNotFoundError:
        st.error("""
        Error: No se pudieron cargar los archivos 'model_rf.joblib' o 
        'training_columns.joblib'. 
        Asegúrese de haber ejecutado 'train_model.py' primero.
        """)
        return None, None

@st.cache_data
def load_data_for_widgets():
    """Carga los datos iniciales para obtener las categorías únicas de los select boxes."""
    try:
        df = pd.read_csv('car_price_prediction.csv')
        # Limpieza mínima necesaria solo para obtener categorías únicas si es necesario
        df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '', regex=False) # [2]
        return df
    except FileNotFoundError:
        return pd.DataFrame() # Retorna vacío si falla

model_rf, training_columns = load_model_resources()
df_raw = load_data_for_widgets()

# -----------------------------
# 2. INTERFAZ DE ENTRADA DE USUARIO (WIDGETS)
# -----------------------------

st.sidebar.header("Variables de Predicción") # [1]

if model_rf is not None and not df_raw.empty:
    
    # Inputs Numéricos
    # Prod. year (Basado en el rango de los datos si están disponibles)
    min_year = int(df_raw['Prod. year'].min()) if not df_raw.empty else 1990
    max_year = int(df_raw['Prod. year'].max()) if not df_raw.empty else 2020

    prod_year = st.sidebar.number_input(
        "Año de Producción (Prod. year):", 
        min_value=min_year,
        max_value=max_year,
        value=2015,
        step=1
    )
    
    # Inputs Categóricos (Usando st.selectbox) [1]
    
    # Manufacturer
    manufacturer = st.sidebar.selectbox(
        "Fabricante (Manufacturer):",
        options=df_raw["Manufacturer"].unique()
    )
    
    # Model
    model_car = st.sidebar.selectbox(
        "Modelo (Model):",
        options=df_raw["Model"].unique()
    )
    
    # Category
    category = st.sidebar.selectbox(
        "Categoría:",
        options=df_raw["Category"].unique()
    )
    
    # Fuel Type
    fuel_type = st.sidebar.selectbox(
        "Tipo de Combustible (Fuel type):",
        options=df_raw["Fuel type"].unique()
    )
    
    # Gear Box Type
    gear_box_type = st.sidebar.selectbox(
        "Tipo de Caja de Cambios (Gear box type):",
        options=df_raw["Gear box type"].unique()
    )

    # Input de Feature Engineered (Is_Turbo)
    # Se relaciona con 'Engine volume' que contiene 'Turbo' [2]
    is_turbo = st.sidebar.checkbox("¿Es Turbo?", value=False)
    
    # Botón para activar la predicción [3]
    if st.button("Obtener Predicción"):
        
        # -----------------------------
        # 3. LÓGICA DE PREDICCIÓN Y PREPROCESAMIENTO DE ENTRADA
        # -----------------------------
        
        # A. Crear el DataFrame de entrada con las variables del usuario
        data_input = {
            'Prod. year': [prod_year],
            'Manufacturer': [manufacturer],
            'Model': [model_car],
            'Category': [category],
            'Fuel type': [fuel_type],
            'Gear box type': [gear_box_type]
        }
        input_df = pd.DataFrame(data_input)
        
        # B. Aplicar la Codificación One-Hot a las columnas categóricas seleccionadas [4, 5]
        categorical_cols_app = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type']
        input_dummies = pd.get_dummies(input_df[categorical_cols_app], drop_first=True)
        
        # C. Preparar el DataFrame final X_pred usando las training_columns
        
        # Crear un DataFrame vacío con todas las columnas de entrenamiento y llenarlo con ceros
        X_pred = pd.DataFrame(data=0, index='', columns=training_columns)
        
        # Mapear valores numéricos
        X_pred['Prod. year'] = prod_year
        
        # Mapear característica Is_Turbo [2]
        # Esta característica no pasó por get_dummies en la selección original [4-6]
        if 'Is_Turbo' in training_columns: 
             X_pred['Is_Turbo'] = 1 if is_turbo else 0
             
        # Mapear variables categóricas (One-Hot)
        for col in input_dummies.columns:
            # Si la columna (ej: 'Manufacturer_BMW') existe en las columnas de entrenamiento
            if col in training_columns:
                X_pred[col] = input_dummies[col].iloc

        # D. Realizar la Predicción
        try:
            prediction = model_rf.predict(X_pred) # [7]
            
            # E. Mostrar el resultado
            st.subheader("Resultado de la Predicción")
            st.success(f"**Predicción del precio del vehículo (USD): ${prediction:,.2f}**") # [7]
            
            st.markdown("---")
            st.caption("Nota: La predicción se realiza con el modelo Random Forest ajustado en el script de entrenamiento, usando las variables de entrada que cumplen con el preprocesamiento One-Hot Encoding.")

        except ValueError as e:
            st.error(f"Error al realizar la predicción. Asegúrese que el formato de entrada es correcto: {e}")

else:
    st.warning("No se pudo iniciar la aplicación. Revise los mensajes de error anteriores.")
