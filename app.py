import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. CONFIGURACIÓN
st.set_page_config(page_title="Simulador ML Interactiva", layout="wide")
st.title("📊 Simulador de Predicción en Tiempo Real")
st.markdown("Mueve los controles laterales para ver cómo cambian las predicciones instantáneamente.")

# 2. CARGA DE DATOS (Cache para velocidad)
@st.cache_data
def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target
    return df, data.feature_names

df, features = load_data()

# 3. SIDEBAR INTERACTIVO
st.sidebar.header("⚙️ Configuración del Modelo")
epocas = st.sidebar.slider("Épocas (RNA)", 10, 500, 100)

st.sidebar.divider()
st.sidebar.header("🖱️ Variables de Entrada")
input_data = {}
for col in features:
    opciones = np.linspace(df[col].min(), df[col].max(), num=15)
    opciones_formateadas = [round(float(x), 2) for x in opciones]
    input_data[col] = st.sidebar.selectbox(f"{col}", options=opciones_formateadas, index=7)

# 4. PREPROCESAMIENTO RÁPIDO
X = df[features]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
user_input_scaled = scaler.transform(pd.DataFrame([input_data]))

# 5. ENTRENAMIENTO AUTOMÁTICO (Sin botón)
# Nota: Usamos st.cache_resource para el modelo si no queremos que re-entrene TODO cada segundo, 
# pero para este ejercicio de épocas, lo dejamos fluir.

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epocas, random_state=42)
mlp.fit(X_train_scaled, y_train)
pred_mlp = mlp.predict(user_input_scaled)[0]

rf = RandomForestRegressor(n_estimators=50, random_state=42) # Reducimos n_estimators para velocidad
rf.fit(X_train, y_train)
pred_rf = rf.predict(pd.DataFrame([input_data]))[0]

# 6. VISUALIZACIÓN DE RESULTADOS
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🤖 Resultados de los Modelos")
    st.metric("Predicción RNA", f"{pred_mlp:.3f}")
    st.metric("Predicción Random Forest", f"{pred_rf:.3f}")
    
    # Diferencia porcentual
    dif = abs(pred_mlp - pred_rf)
    st.write(f"**Diferencia entre modelos:** {dif:.4f}")

with col2:
    st.subheader("📈 Comparativa Visual")
    fig, ax = plt.subplots()
    modelos = ['Red Neuronal', 'Random Forest']
    predicciones = [pred_mlp, pred_rf]
    
    sns.barplot(x=modelos, y=predicciones, palette='viridis', ax=ax)
    ax.set_ylabel('Precio Predicho (Unidades de 100k)')
    ax.set_title('Comparación de Predicción Actual')
    st.pyplot(fig)

# 7. GRÁFICO DE IMPORTANCIA DE VARIABLES (Extra para el taller)
st.divider()
st.subheader("📌 Importancia de las Variables (Random Forest)")
importancias = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
fig2, ax2 = plt.subplots()
importancias.plot(kind='barh', color='skyblue', ax=ax2)
st.pyplot(fig2)