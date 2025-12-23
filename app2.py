import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
st.set_page_config(page_title="Predicción de Autos", layout="wide")
st.title("🚗 Predicción del Precio de Carros con Random Forest")
st.subheader("Utilice la barra lateral para ingresar las características del vehículo.")
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FF4B4B; /* Rojo */
        color: white;
        width: 100%; 
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF2B2B;
        color: white;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 1. FUNCIÓN DE CARGA Y ENTRENAMIENTO CACHEADO
# Se usa st.cache_resource para entrenar solo una vez en el servidor
# ----------------------------------------------------------------------

@st.cache_resource 
def load_and_train_model():
    """Carga los datos, aplica el preprocesamiento completo, entrena el modelo 
    y retorna el modelo ajustado, las columnas de entrenamiento, la grafica de importancia y el dataframe base."""
    
    mensaje_carga = st.empty()
    mensaje_carga.info("Iniciando carga y entrenamiento del modelo (Esto solo ocurre la primera vez)...")
    
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
    
    mensaje_carga.empty()
    st.success("✅ Modelo listo para usar.")
    importancias = pd.Series(model_rf.feature_importances_, index=training_columns)
    
    # Retornamos el modelo, las columnas de entrenamiento , el gáfico de importancias y el DataFrame seleccionado (para widgets)
    return model_rf, training_columns, df_selected , importancias

# Cargar/Entrenar el modelo
model_rf, training_columns, df_selected, importancias_raw = load_and_train_model()

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
    
    #Creamos un df que solo tenga valores para el fabricante seleccionado.
    df_modelos_filtrados = df_selected[df_selected["Manufacturer"] == manufacturer]
    
    model_car = st.sidebar.selectbox(
        "Modelo (Model):",
        options=df_modelos_filtrados["Model"].unique()
    )
    
    #Creamos un df que solo tenga valores para el modelo seleccionado.
    df_categorias_filtradas = df_modelos_filtrados[df_modelos_filtrados["Model"] == model_car]
    
    category = st.sidebar.selectbox(
        "Categoría:",
        options=df_categorias_filtradas["Category"].unique()
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
    
    if st.sidebar.button("Obtener Predicción"):
        
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
        
        # 2. Preparar el DataFrame X_pred con CEROS usando las columnas exactas del modelo
        # Esto asegura que el modelo reciba la estructura que espera.
        X_pred = pd.DataFrame(data=0, index=[0], columns=training_columns) 
        
        # 3. Asignar valor numérico del año
        X_pred['Prod. year'] = prod_year
        
        # 4. Asignar Is_Turbo
        if 'Is_Turbo' in training_columns: 
             X_pred['Is_Turbo'] = 1 if is_turbo else 0
             
        # 5. MAPEADO DINÁMICO
        
        columnas_categoricas = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type']
        
        for col_nombre in columnas_categoricas:
            valor_elegido = str(input_df[col_nombre].iloc[0])
            # Construimos el nombre de la columna: "NombreColumna_Valor"
            # Ejemplo: "Manufacturer_NISSAN"
            col_con_prefijo = f"{col_nombre}_{valor_elegido}"
            
            if col_con_prefijo in training_columns:
                X_pred[col_con_prefijo] = 1

        # 6. Realizar la Predicción
        try:
            prediction = model_rf.predict(X_pred)[0]
            
            st.markdown("---")
            # --- PARTE SUPERIOR: CUADROS CENTRADOS ---
            col_izq, col_der = st.columns(2)
            
            with col_izq:
                # Cuadro Azul para el Precio
                st.markdown("""
                    <div style="background-color:#1E3A8A; color:white; padding:25px; border-radius:15px; text-align:center;">
                        <h3 style="color:white; margin:0;">Precio Predicho</h3>
                        <h1 style="color:white; margin:10px 0;">${:,.2f} USD</h1>
                    </div>
                """.format(prediction), unsafe_allow_html=True)

            with col_der:
                # Cuadro Gris con el Resumen (Características)
                st.markdown(f"""
                    <div style="background-color:#374151; color:white; padding:20px; border-radius:15px; font-size: 14px;">
                        <h4 style="color:white; margin-top:0;">Resumen del Vehículo</h4>
                        <hr style="margin:10px 0; border:0.5px solid #4B5563;">
                        <b>Fabricante:</b> {manufacturer}<br>
                        <b>Modelo:</b> {model_car}<br>
                        <b>Año:</b> {prod_year}<br>
                        <b>Categoría:</b> {category}<br>
                        <b>Combustible:</b> {fuel_type}<br>
                        <b>Transmisión:</b> {gear_box_type}
                    </div>
                """, unsafe_allow_html=True)

            # --- PARTE INFERIOR: GRÁFICO DE BARRAS (Debajo de los cuadros) ---
            st.write("##") # Espacio
            st.subheader("📊 Importancia de Variables en esta Predicción")
            
            # Procesamos las importancias para que se vean bien (Top 7)
            import pandas as pd
            import plotly.express as px
            
            # Limpiamos los nombres de las columnas para el gráfico
            feat_imp = importancias_raw.sort_values(ascending=False).head(7)
            
            fig = px.bar(
                x=feat_imp.values,
                y=feat_imp.index,
                orientation='h',
                color='Variable',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={'x': 'Influencia en el Precio', 'y': 'Variable'}
            )
            fig.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
