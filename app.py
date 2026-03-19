import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Prediccción ML", layout="wide")
st.title("🧠 Simulador de Entrenamiento: RNA vs Random Forest")

# 2. CARGA Y PREPROCESAMIENTO DE DATOS (Del Cuaderno a Streamlit)
@st.cache_data
def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target
    return df, data.feature_names

df, features = load_data()

# 3. SIDEBAR - PARÁMETROS DE SIMULACIÓN
st.sidebar.header("⚙️ Configuración del Modelo")
epocas = st.sidebar.slider("Selecciona las Épocas (Iteraciones RNA)", 10, 500, 100)
test_size = st.sidebar.slider("Tamaño de prueba (%)", 10, 50, 20) / 100

st.sidebar.divider()
st.sidebar.header("⌨️ Datos para Predicción")
input_data = {}
for col in features:
    input_data[col] = st.sidebar.number_input(f"Valor {col}", value=float(df[col].mean()))

# 4. PROCESAMIENTO
X = df[features]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
user_input_scaled = scaler.transform(pd.DataFrame([input_data]))

# 5. BOTÓN DE ENTRENAMIENTO
if st.button("🔥 Iniciar Entrenamiento y Comparación"):
    
    # Modelo A: Red Neuronal
    with st.spinner('Entrenando Red Neuronal...'):
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epocas, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        pred_mlp = mlp.predict(user_input_scaled)[0]
        r2_mlp = r2_score(y_test, mlp.predict(X_test_scaled))

    # Modelo B: Random Forest
    with st.spinner('Entrenando Random Forest...'):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train) # RF no requiere escalado obligatoriamente
        pred_rf = rf.predict(pd.DataFrame([input_data]))[0]
        r2_rf = r2_score(y_test, rf.predict(X_test))

    # 6. MOSTRAR RESULTADOS
    st.success("¡Modelos entrenados con éxito!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🤖 Red Neuronal")
        st.metric("Predicción", f"{pred_mlp:.4f}")
        st.info(f"Precisión R²: {r2_mlp:.4f}")
        
    with col2:
        st.header("🌲 Random Forest")
        st.metric("Predicción", f"{pred_rf:.4f}")
        st.info(f"Precisión R²: {r2_rf:.4f}")

    # Comparativa de Error
    error_diff = abs(pred_mlp - pred_rf)
    st.write(f"**Diferencia entre modelos:** {error_diff:.4f}")