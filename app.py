# Importaciones necesarias
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os



# Silenciar advertencias específicas
warnings.filterwarnings("ignore", category=UserWarning, message="Found unknown categories in columns")
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score,
    recall_score, f1_score, classification_report, roc_curve,
    auc, precision_recall_curve, average_precision_score
)

# -------------------------------------------------------------------
# 1) Carga de clases personalizadas (engañando a joblib)
# -------------------------------------------------------------------
import utils.transformers
from scripts.extract_url_features import extract_variables_from_url

# Importa todas las clases necesarias desde utils.transformers
page_rank_condition = utils.transformers.page_rank_condition
ratio_digits_url_increment_condition = utils.transformers.ratio_digits_url_increment_condition
nb_subdomains_increment_condition = utils.transformers.nb_subdomains_increment_condition
length_words_raw_increment_condition = utils.transformers.length_words_raw_increment_condition
char_repeat_increment_condition = utils.transformers.char_repeat_increment_condition
shortest_word_host_increment_condition = utils.transformers.shortest_word_host_increment_condition
nb_slash_increment_condition = utils.transformers.nb_slash_increment_condition
longest_word_host_increment_condition = utils.transformers.longest_word_host_increment_condition
avg_word_host_increment_condition = utils.transformers.avg_word_host_increment_condition

# Define las clases personalizadas necesarias
class DropColumns(utils.transformers.DropColumns):
    pass

class AddWeirdColumn(utils.transformers.AddWeirdColumn):
    pass

class EncodeCategorical(utils.transformers.EncodeCategorical):
    pass

class SklearnXGBClassifier(utils.transformers.SklearnXGBClassifier):
    pass

class DataFrameStandardScaler(utils.transformers.DataFrameStandardScaler):
    pass

class RemoveBinaryDuplicates(utils.transformers.RemoveBinaryDuplicates):
    pass

# -------------------------------------------------------------------
# 2) Funciónes avanzada
# -------------------------------------------------------------------
from utils.functions import evaluar_modelo_completo





# -------------------------------------------------------------------
# 3) Función para mostrar métricas sencillas
# -------------------------------------------------------------------
def mostrar_metricas(y_true, y_pred, proba=None):
    acc = (y_pred == y_true).mean()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, proba) if proba is not None and len(np.unique(proba)) > 1 else None

    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precisión:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    if roc is not None:
        st.write(f"**ROC AUC:** {roc:.4f}")
    else:
        st.write("**ROC AUC:** No disponible (proba insuficiente o `predict_proba` no soportado).")

# -------------------------------------------------------------------
# 4) Inicializar session_state si no existe
# -------------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state["model"] = None
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "df_data" not in st.session_state:
    st.session_state["df_data"] = None
if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.5
if "analisis_avanzado" not in st.session_state:
    st.session_state["analisis_avanzado"] = False
if "figuras_analisis_avanzado" not in st.session_state:
    st.session_state["figuras_analisis_avanzado"] = []
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "y_proba" not in st.session_state:
    st.session_state["y_proba"] = None
if "y_pred" not in st.session_state:
    st.session_state["y_pred"] = None

# -------------------------------------------------------------------
# 5) Funciones de carga (modelo & dataset), actualizando session_state
# -------------------------------------------------------------------
def cargar_modelo_pipeline():
    """Carga modelo, pipeline y metadatos a session_state."""
    if st.session_state["model"] is None:
        try:
            st.session_state["model"] = joblib.load("Modelos/mejor_modelo.pkl")
            st.session_state["pipeline"] = joblib.load("Modelos/mejor_pipeline.pkl")
            st.session_state["metadata"] = joblib.load("Modelos/metadatos.pkl")
            st.success("Modelo, pipeline y metadatos cargados correctamente.")

            with st.expander("Ver metadatos del modelo"):
                st.json(st.session_state["metadata"])
        except Exception as e:
            st.error(f"Error al cargar modelo/pipeline: {e}")
    else:
        st.info("El modelo y pipeline ya están cargados en esta sesión.")

def cargar_dataset(ruta):
    """Carga el parquet en df_data."""
    try:
        df = pd.read_parquet(ruta)
        st.session_state["df_data"] = df
        st.success(f"Dataset '{ruta}' cargado con éxito. Shape: {df.shape}")
    except Exception as e:
        st.error(f"Error al cargar dataset '{ruta}': {e}")

# -------------------------------------------------------------------
# 6) Función para descargar el dataset con predicciones
# -------------------------------------------------------------------
def descargar_predicciones(df, key):
    """Permite descargar el dataframe con las predicciones."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Predicciones como CSV",
        data=csv,
        file_name='predicciones_phishing.csv',
        mime='text/csv',
        key=key  # Asignar el key único aquí
    )

#---------------------------------------------------------------------------------------------------
# 7) Funcion para detectar el protocolo
#---------------------------------------------------------------------------------------------------
import requests

def detect_protocol(url):
    """
    Intenta determinar si la URL funciona con HTTPS o HTTP.
    - Si el usuario ya incluye 'http://' o 'https://', la devuelve tal cual.
    - Si no, prueba primero con HTTPS y luego con HTTP.
    - Si ambas fallan, devuelve None.
    """
    if url.startswith(("http://", "https://")):
        return url  # ✅ Si ya tiene protocolo, no tocamos nada

    url_https = "https://" + url
    url_http = "http://" + url

    try:
        # 🔍 Intentamos acceder con HTTPS primero
        response = requests.head(url_https, allow_redirects=True, timeout=3)
        if response.status_code < 400:
            return url_https  # ✅ Si funciona con HTTPS, lo usamos
    except requests.RequestException:
        pass

    try:
        # 🔍 Si HTTPS falla, probamos con HTTP
        response = requests.head(url_http, allow_redirects=True, timeout=3)
        if response.status_code < 400:
            return url_http  # ✅ Si funciona con HTTP, lo usamos
    except requests.RequestException:
        pass

    return None  # ❌ Si ambas fallan, la URL no es accesible

# ---------------- PERSISTENCIA DE PESTAÑA ----------------

# Si la clave "active_tab" no está en session_state, inicializarla
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Check URL"



# -------------------------------------------------------------------
# 8) Lógica principal
# -------------------------------------------------------------------
def main():
    st.title("Sistema de Detección de URLs Phishing :lock:")
    st.write("Bienvenido a la demo interactiva de detección de phishing. Usa la **barra lateral** para cargar el modelo y el dataset, y ajustar el umbral.")
    
    # Barra lateral:
    st.sidebar.subheader("Cargar Modelo y Pipeline")
    if st.sidebar.button("Cargar modelo/pipeline"):
        cargar_modelo_pipeline()
    
    # Mostrar estado del modelo
    if st.session_state["model"] is not None:
        st.sidebar.success("Modelo & pipeline cargados en session_state")
    else:
        st.sidebar.info("Modelo no cargado aún.")
    
    # Elegir dataset
    st.sidebar.subheader("Cargar Dataset")
    dataset_option = st.sidebar.selectbox("Selecciona el dataset a cargar", ["Ninguno", "Data/train.parquet", "Data/test.parquet"])
    if st.sidebar.button("Cargar Dataset"):
        if dataset_option == "Ninguno":
            st.warning("No se ha seleccionado ningún dataset.")
        else:
            cargar_dataset(dataset_option)
    
    # Mostrar estado del dataset
    if st.session_state["df_data"] is not None:
        st.sidebar.success("Dataset cargado en session_state")
    else:
        st.sidebar.info("Sin dataset cargado.")
    
    # Cuadro de texto para Threshold
    thr_input = st.sidebar.text_input("Umbral de decisión (Threshold)", value=str(st.session_state["threshold"]))
    try:
        thr = float(thr_input)
        if thr < 0.0 or thr > 1.0:
            st.sidebar.error("El umbral debe estar entre 0 y 1.")
            thr = st.session_state["threshold"]
        else:
            st.session_state["threshold"] = thr
    except ValueError:
        st.sidebar.error("Por favor, ingresa un valor numérico válido para el umbral (entre 0 y 1).")
        thr = st.session_state["threshold"]
    
    # Crear tabs
    # Crear pestañas con estado persistente

    tab_selected = st.session_state["active_tab"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Vista de Datos", "📊 Predicción & Métricas", "📈 Análisis Avanzado", "🆚 Comparativa Modelos", "🔍 Predicción por URL"])
    
    
    # ---------------- TAB 1: Vista de Datos ----------------
    with tab1:
        st.subheader("Vista Preliminar del Dataset")
        df_data = st.session_state.get("df_data", None)
        if df_data is not None:
            # Mostrar nombres de las columnas
            st.write("**Nombres de las columnas en el dataset:**")
            st.write(df_data.columns.tolist())

            # Definir las columnas a mostrar
            cols_to_display = []
            if 'status' in df_data.columns and 'url' in df_data.columns:
                cols_to_display = ['url', 'status']
            elif 'url' in df_data.columns:
                cols_to_display = ['url']

            if 'ID' in df_data.columns:
                cols_to_display = ['ID'] + cols_to_display

            # Mostrar las columnas seleccionadas si existen
            missing_cols = [col for col in cols_to_display if col not in df_data.columns]
            if not missing_cols:
                st.write("**Vista Preliminar de las Predicciones:**")
                st.dataframe(df_data[cols_to_display].head(50))  # Mostrar las primeras 50 filas
                st.write(f"**Shape:** {df_data.shape[0]:,} filas x {df_data.shape[1]} columnas.")
            else:
                st.warning(f"No se encontraron las columnas: {', '.join(missing_cols)} en el dataset.")
        else:
            st.info("Selecciona y carga un dataset en la barra lateral para ver sus datos aquí.")
    
    # ---------------- TAB 2: Predicción & Métricas ----------------
    with tab2:
        st.subheader("Generar Predicciones y Ver Métricas Básicas")
        model = st.session_state.get("model", None)
        pipeline = st.session_state.get("pipeline", None)
        df_data = st.session_state.get("df_data", None)
        threshold = st.session_state.get("threshold", 0.5)

        if df_data is not None and model is not None and pipeline is not None:
            # Crear una copia para el modelo, eliminando 'ID' si existe
            df_model = df_data.copy()
            if "ID" in df_model.columns:
                df_model.drop(columns=["ID"], inplace=True, errors="ignore")

            # Revisar si existe 'status'
            if "status" in df_model.columns:
                y_test = df_model["status"].values
                X_test = df_model.drop(columns=["status"], errors="ignore")
            else:
                y_test = None
                X_test = df_model

            # Transformar los datos usando el pipeline **solo las características predictoras**
            try:
                X_test_proc = pipeline.transform(X_test)
                if isinstance(X_test_proc, np.ndarray):
                    try:
                        X_test_proc = pd.DataFrame(X_test_proc, columns=pipeline.get_feature_names_out())
                    except AttributeError:
                        st.warning("El pipeline no tiene método `get_feature_names_out()`. Usando nombres genéricos de features.")
                        X_test_proc = pd.DataFrame(X_test_proc, columns=[f"Feature {i}" for i in range(X_test_proc.shape[1])])
                st.session_state["X_test_proc"] = X_test_proc  # Guardar para análisis
            except Exception as e:
                st.error(f"Error al transformar los datos: {e}")
                X_test_proc = None

            if X_test_proc is not None:
                # Predicción con umbral
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test_proc)[:, 1]
                        y_pred = (y_proba >= threshold).astype(int)
                    else:
                        st.warning("El modelo no soporta `predict_proba`. Se usará `predict` para generar las predicciones.")
                        y_pred = model.predict(X_test_proc)
                        y_proba = None
                except Exception as e:
                    st.error(f"Error en predict_proba/predict: {e}")
                    y_pred = None
                    y_proba = None

                if y_pred is not None:
                    # Añadir las predicciones al dataframe de display
                    df_display = df_data.copy()

                    # Verificar qué columnas existen antes de usarlas
                    cols_to_display = []
                    for col in ['ID', 'url', 'status']:
                        if col in df_display.columns:
                            cols_to_display.append(col)

                    # Añadir 'Predicción' y 'Proba' al dataframe
                    df_display["Predicción"] = y_pred
                    df_display["Proba"] = y_proba if y_proba is not None else None  # Pandas maneja `None` como NaN

                    # Mostrar las predicciones
                    st.write("**Todas las Predicciones:**")
                    st.dataframe(df_display[cols_to_display + ["Predicción", "Proba"]].head(50))

                    # Botón para descargar todas las predicciones
                    descargar_predicciones(df_display, key='download_all_predictions')

                    # Tabla de predicciones erróneas
                    if y_test is not None:
                        df_errors = df_display[df_display["status"] != df_display["Predicción"]]
                        st.write("**Predicciones Erróneas:**")
                        if df_errors.empty:
                            st.success("¡No hay predicciones erróneas!")
                        else:
                            # Añadir 'Proba' si no está presente
                            if 'Proba' not in df_errors.columns and y_proba is not None:
                                df_errors["Proba"] = y_proba[df_errors.index]
                            st.dataframe(df_errors[cols_to_display + ["Predicción", "Proba"]].head(50))  # Mostrar las primeras 50 filas
                            st.write(f"**Número Total de Errores:** {df_errors.shape[0]:,}")
                            # **Botón para descargar solo las predicciones erróneas**
                            descargar_predicciones(df_errors, key='download_error_predictions')

                            st.write(f"**Umbral Actual:** {threshold:.2f}")
                            mostrar_metricas(y_test, y_pred, proba=y_proba)

                            # Matriz de confusión (tamaño reducido)
                            cm = confusion_matrix(y_test, y_pred)
                            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))  # Tamaño reducido
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
                            ax_cm.set_xlabel("Predicción")
                            ax_cm.set_ylabel("Real")
                            ax_cm.set_title("Matriz de Confusión")
                            fig_cm.tight_layout()
                            st.pyplot(fig_cm)

                            # Gráfico de Barras para 'is_weird' basado en Predicción
                            if 'is_weird' in X_test_proc.columns:
                                st.write("**Proporción de Phishing en `is_weird`**")
                                df_weird = pd.DataFrame({
                                    'is_weird': X_test_proc['is_weird'],
                                    'Phishing': y_pred  # Usar predicciones en lugar de 'status'
                                })
                                proportion_weird = df_weird.groupby('is_weird')['Phishing'].mean().reset_index()
                                proportion_weird['Phishing'] = proportion_weird['Phishing'] * 100  # Convertir a porcentaje
                                
                                fig_weird, ax_weird = plt.subplots(figsize=(8, 6))
                                sns.barplot(data=proportion_weird, x='is_weird', y='Phishing', ax=ax_weird)
                                ax_weird.set_title("Proporción de Phishing por `is_weird`")
                                ax_weird.set_xlabel("is_weird")
                                ax_weird.set_ylabel("Proporción de Phishing (%)")
                                for index, row in proportion_weird.iterrows():
                                    ax_weird.text(index, row.Phishing + 0.5, f"{row.Phishing:.2f}%", color='black', ha="center")
                                fig_weird.tight_layout()
                                st.pyplot(fig_weird)
                            else:
                                st.warning("La columna `is_weird` no está presente en los datos transformados.")

                            # **Agregar Gráfica de Número de Fallos por Probabilidad**
                            if y_proba is not None:
                                st.write("**Número de Fallos por Probabilidad de Predicción**")
                                fig_failures_prob, ax_failures_prob = plt.subplots(figsize=(10, 6))
                                sns.histplot(df_errors['Proba'], bins=20, kde=False, ax=ax_failures_prob, color='red')
                                ax_failures_prob.set_title("Distribución de Probabilidades en Fallos")
                                ax_failures_prob.set_xlabel("Probabilidad de Phishing")
                                ax_failures_prob.set_ylabel("Número de Fallos")
                                fig_failures_prob.tight_layout()
                                st.pyplot(fig_failures_prob)
                            else:
                                st.info("No se puede generar la gráfica de fallos por probabilidad porque `predict_proba` no está disponible.")
                    else:
                        st.info("El dataset no contiene la columna 'status'. Se muestran solo predicciones.")
        else:
            st.info("Debes cargar un dataset y haber cargado el modelo/pipeline para ver métricas.")
    
    # ---------------- TAB 3: Análisis Avanzado ----------------
    with tab3:
        st.subheader("Análisis Avanzado del Modelo (Errores y Fortalezas)")
        model = st.session_state.get("model", None)
        pipeline = st.session_state.get("pipeline", None)
        df_data = st.session_state.get("df_data", None)
        threshold = st.session_state.get("threshold", 0.5)

        if df_data is not None and model is not None and pipeline is not None:
            if "status" not in df_data.columns:
                st.warning("El dataset no contiene la columna 'status'. No se puede realizar el análisis avanzado.")
            else:
                # Crear una copia para el análisis, eliminando 'ID' si existe
                df_model_analysis = df_data.copy()
                if "ID" in df_model_analysis.columns:
                    df_model_analysis.drop(columns=["ID"], inplace=True, errors="ignore")

                # Separar características y variable objetivo
                y_test = df_model_analysis["status"].values
                X_test = df_model_analysis.drop(columns=["status"], errors="ignore")

                # Transformar solo las características predictoras
                try:
                    X_test_proc = pipeline.transform(X_test)
                    if isinstance(X_test_proc, np.ndarray):
                        try:
                            X_test_proc = pd.DataFrame(X_test_proc, columns=pipeline.get_feature_names_out())
                        except AttributeError:
                            st.warning("El pipeline no tiene método `get_feature_names_out()`. Usando nombres genéricos de features.")
                            X_test_proc = pd.DataFrame(X_test_proc, columns=[f"Feature {i}" for i in range(X_test_proc.shape[1])])
                    st.session_state["X_test_proc"] = X_test_proc  # Guardar para análisis
                except Exception as e:
                    st.error(f"Error al transformar los datos: {e}")
                    X_test_proc = None

                if X_test_proc is not None:
                    # Predicciones
                    try:
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test_proc)[:, 1]
                            y_pred = (y_proba >= threshold).astype(int)
                        else:
                            st.warning("El modelo no soporta `predict_proba`. Se usará `predict` para generar las predicciones.")
                            y_pred = model.predict(X_test_proc)
                            y_proba = None
                    except Exception as e:
                        st.error(f"Error en predict_proba/predict: {e}")
                        y_pred = None
                        y_proba = None

                    if y_pred is not None:
                        # Llamar a la función de evaluación completa
                        figs = evaluar_modelo_completo(model, X_test_proc, y_test, model_name="Phishing Demo")
                        st.session_state["figuras_analisis_avanzado"] = figs
                        st.session_state["y_test"] = y_test
                        st.session_state["y_proba"] = y_proba
                        st.session_state["y_pred"] = y_pred
                        st.session_state["analisis_avanzado"] = True
                        st.success("Análisis avanzado calculado y almacenado.")

                        # Mostrar los gráficos almacenados
                        if st.session_state["analisis_avanzado"]:
                            figs = st.session_state.get("figuras_analisis_avanzado", [])
                            y_test = st.session_state.get("y_test", None)
                            y_proba = st.session_state.get("y_proba", None)
                            y_pred = st.session_state.get("y_pred", None)

                            if figs:
                                for i, fig in enumerate(figs, start=1):
                                    st.write(f"**Gráfico {i}**:")
                                    st.pyplot(fig)

                                # Añadir análisis adicional: URLs donde el modelo falla
                                st.markdown("---")
                                st.subheader("Análisis de URLs donde el Modelo Falla")

                                # Crear dataframe con predicciones
                                df_analysis = df_data.copy()
                                if "ID" in df_analysis.columns:
                                    df_analysis = df_analysis[['ID', 'url', 'status']].copy()
                                else:
                                    df_analysis = df_analysis[['url', 'status']].copy()

                                # Añadir las predicciones y probabilidades
                                df_analysis["Predicción"] = y_pred
                                if y_proba is not None:
                                    df_analysis["Proba"] = y_proba
                                else:
                                    df_analysis["Proba"] = np.nan
                                df_analysis["Correcto"] = df_analysis["status"] == df_analysis["Predicción"]

                                # Seleccionar características numéricas y categóricas disponibles
                                available_features = X_test.select_dtypes(include=[np.number, 'category']).columns.tolist()
                                # Añadir 'is_weird' si está disponible
                                if 'is_weird' in X_test_proc.columns and 'is_weird' not in available_features:
                                    available_features.append('is_weird')

                                if not available_features:
                                    st.warning("No hay características numéricas o categóricas disponibles para el análisis.")
                                else:
                                    selected_feature = st.selectbox("Selecciona una característica para analizar", available_features)

                                    if selected_feature:
                                        # Añadir la característica al dataframe de análisis
                                        if selected_feature in df_data.columns:
                                            df_analysis[selected_feature] = df_data[selected_feature]
                                        elif selected_feature in X_test_proc.columns:
                                            df_analysis[selected_feature] = X_test_proc[selected_feature]
                                        else:
                                            st.warning(f"La característica seleccionada '{selected_feature}' no está disponible.")
                                            df_analysis[selected_feature] = np.nan

                                        # Determinar si es numérica o categórica
                                        if pd.api.types.is_numeric_dtype(df_analysis[selected_feature]):
                                            unique_values = df_analysis[selected_feature].nunique()
                                            if unique_values > 10:
                                                # Convertir a float32 para evitar el error
                                                df_analysis[selected_feature] = df_analysis[selected_feature].astype(np.float32)
                                                
                                                # Binarizar la variable en categorías (por ejemplo, usando cuantiles)
                                                num_bins = st.slider("Selecciona el número de bins para la característica continua", min_value=2, max_value=10, value=5)
                                                try:
                                                    df_analysis[f"{selected_feature}_binned"] = pd.qcut(df_analysis[selected_feature], q=num_bins, duplicates='drop')
                                                except ValueError:
                                                    st.warning("No se pudo binarizar la característica seleccionada. Intentando con menos bins.")
                                                    df_analysis[f"{selected_feature}_binned"] = pd.qcut(df_analysis[selected_feature], q=num_bins-1, duplicates='drop')
                                                feature_to_plot = f"{selected_feature}_binned"
                                            else:
                                                # Usar la característica directamente si tiene pocos valores únicos
                                                feature_to_plot = selected_feature
                                        else:
                                            feature_to_plot = selected_feature

                                        # Calcular las proporciones de aciertos y errores por categoría
                                        df_grouped = df_analysis.groupby(feature_to_plot)["Correcto"].value_counts(normalize=True).unstack(fill_value=0)

                                        # Renombrar las columnas para claridad
                                        if 'False' in df_grouped.columns and 'True' in df_grouped.columns:
                                            df_grouped.columns = ["Errores", "Aciertos"]
                                        elif 'True' in df_grouped.columns:
                                            df_grouped.columns = ["Aciertos"]
                                        elif 'False' in df_grouped.columns:
                                            df_grouped.columns = ["Errores"]

                                        # Gráfico de barras apiladas con etiquetas de proporción
                                        fig_grouped, ax_grouped = plt.subplots(figsize=(12, 6))
                                        df_grouped.plot(kind='bar', stacked=True, color=['red', 'green'], ax=ax_grouped)

                                        ax_grouped.set_title(f"Proporción de Aciertos y Errores por '{feature_to_plot}'")
                                        ax_grouped.set_xlabel(feature_to_plot)
                                        ax_grouped.set_ylabel("Proporción")
                                        ax_grouped.legend(["Errores", "Aciertos"], loc='upper right')

                                        # Añadir etiquetas con las proporciones
                                        for container in ax_grouped.containers:
                                            ax_grouped.bar_label(container, fmt="%.2f", label_type='center')

                                        fig_grouped.tight_layout()
                                        st.pyplot(fig_grouped)

                                        st.markdown("""
                                        **Interpretación:**
                                        - **Aciertos (Verde):** Proporción de URLs clasificadas correctamente en cada categoría de la característica seleccionada.
                                        - **Errores (Rojo):** Proporción de URLs clasificadas incorrectamente en cada categoría de la característica seleccionada.
                                        - Observa las diferencias en las proporciones para identificar patrones que podrían ayudar a mejorar el modelo.
                                        """)

                                        # Mostrar proporciones globales
                                        st.write(f"**Proporción de Aciertos:** {np.mean(y_test == y_pred) * 100:.2f}%")
                                        st.write(f"**Proporción de Errores:** {100 - np.mean(y_test == y_pred) * 100:.2f}%")
        else:
            st.info("Debes cargar un dataset y haber cargado el modelo/pipeline para realizar el análisis avanzado.")

    # ---------------- TAB 4: Comparativa de Modelos ----------------
    with tab4:
        st.subheader("Comparación de Varios Modelos")
        st.write("Sube un CSV con métricas de distintos modelos para compararlos.")
        comp_file = st.file_uploader("Subir CSV de Comparativa", key="comp", type=["csv"])
        if comp_file is not None:
            try:
                df_comp = pd.read_csv(comp_file)
                st.dataframe(df_comp)

                # Mostrar información de las columnas
                st.markdown("**Descripción de las Columnas:**")
                st.write(df_comp.dtypes)

                # Verificar columnas necesarias
                required_cols = {"Modelo", "Prueba", "F1-Validation"}
                if required_cols.issubset(df_comp.columns):
                    # Mostrar valores únicos de "Prueba" para clarificación
                    st.markdown("**Valores Únicos de 'Prueba':**")
                    st.write(df_comp["Prueba"].unique())

                    # Gráfico de barras con etiquetas para F1-Validation
                    fig_comp, ax_comp = plt.subplots(figsize=(14, 8))
                    sns.barplot(data=df_comp, x="Modelo", y="F1-Validation", hue="Prueba", ax=ax_comp)
                    ax_comp.set_title("Comparativa de F1-Validation")
                    ax_comp.set_xlabel("Modelo")
                    ax_comp.set_ylabel("F1-Validation")

                    # Añadir etiquetas con los valores
                    for container in ax_comp.containers:
                        ax_comp.bar_label(container, fmt="%.3f", label_type='edge')

                    # Ajustar la leyenda para que no tape el gráfico
                    ax_comp.legend(title="Prueba", loc='upper left', bbox_to_anchor=(1.0, 1))
                    fig_comp.tight_layout()
                    st.pyplot(fig_comp)

                    # Gráfico adicional: Scatter plot de F1-Validation vs F1-Test si existe
                    if {"F1-Test"}.issubset(df_comp.columns):
                        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
                        sns.scatterplot(data=df_comp, x="F1-Validation", y="F1-Test", hue="Modelo", style="Prueba", s=100, ax=ax_scatter)
                        ax_scatter.set_title("F1-Validation vs F1-Test por Modelo y Prueba")
                        ax_scatter.set_xlabel("F1-Validation")
                        ax_scatter.set_ylabel("F1-Test")

                        # Añadir anotaciones para mejor claridad
                        for i in range(df_comp.shape[0]):
                            ax_scatter.text(df_comp["F1-Validation"].iloc[i]+0.001, df_comp["F1-Test"].iloc[i]+0.001,
                                            df_comp["Modelo"].iloc[i], horizontalalignment='left', size='small', color='black', weight='semibold')

                        ax_scatter.legend(title="Prueba", loc='upper left', bbox_to_anchor=(1.0, 1))
                        fig_scatter.tight_layout()
                        st.pyplot(fig_scatter)

                    # Gráfico adicional 1: Overfitting (Comparación de métricas entre Train, Validation y Test)
                    if {"F1-Train", "F1-Validation", "F1-Test"}.issubset(df_comp.columns):
                        fig_overfit, ax_overfit = plt.subplots(figsize=(14, 8))
                        df_overfit = df_comp.melt(id_vars=["Modelo", "Prueba"], value_vars=["F1-Train", "F1-Validation", "F1-Test"], var_name="Conjunto", value_name="F1-Score")
                        sns.barplot(data=df_overfit, x="Modelo", y="F1-Score", hue="Conjunto", ax=ax_overfit)
                        ax_overfit.set_title("Comparación de F1-Score entre Train, Validation y Test")
                        ax_overfit.set_xlabel("Modelo")
                        ax_overfit.set_ylabel("F1-Score")

                        # Añadir etiquetas con los valores
                        for container in ax_overfit.containers:
                            ax_overfit.bar_label(container, fmt="%.3f", label_type='edge')

                        ax_overfit.legend(title="Conjunto", loc='upper left', bbox_to_anchor=(1.0, 1))
                        fig_overfit.tight_layout()
                        st.pyplot(fig_overfit)

                    # Gráfico adicional 2: Intervalos de Confianza (Boxplot de métricas)
                    metrics = ["Accuracy", "Precision", "Recall", "F1-Validation", "F1-Test", "F1-Train"]
                    available_metrics = [metric for metric in metrics if metric in df_comp.columns]
                    if available_metrics:
                        fig_confidence, ax_confidence = plt.subplots(figsize=(14, 8))
                        df_melt = df_comp.melt(id_vars=["Modelo", "Prueba"], value_vars=available_metrics, var_name="Métrica", value_name="Valor")
                        sns.boxplot(data=df_melt, x="Modelo", y="Valor", hue="Métrica", ax=ax_confidence)
                        ax_confidence.set_title("Intervalos de Confianza de las Métricas por Modelo")
                        ax_confidence.set_xlabel("Modelo")
                        ax_confidence.set_ylabel("Valor de la Métrica")

                        # Añadir etiquetas con los valores
                        ax_confidence.legend(title="Métrica", loc='upper left', bbox_to_anchor=(1.0, 1))
                        fig_confidence.tight_layout()
                        st.pyplot(fig_confidence)

                    # Gráfico adicional 3: Diferencia entre Test y Train
                    if {"F1-Train", "F1-Test"}.issubset(df_comp.columns):
                        df_overfit_diff = df_comp.copy()
                        df_overfit_diff["Diferencia"] = abs(df_overfit_diff["F1-Test"] - df_overfit_diff["F1-Train"])


                        fig_diff, ax_diff = plt.subplots(figsize=(14, 8))
                        sns.barplot(data=df_overfit_diff, x="Modelo", y="Diferencia", hue="Prueba", ax=ax_diff)
                        ax_diff.set_title("Diferencia de F1-Score entre Test y Train")
                        ax_diff.set_xlabel("Modelo")
                        ax_diff.set_ylabel("Diferencia de F1-Score")

                        # Añadir etiquetas con los valores
                        for container in ax_diff.containers:
                            ax_diff.bar_label(container, fmt="%.3f", label_type='edge')

                        ax_diff.legend(title="Prueba", loc='upper left', bbox_to_anchor=(1.0, 1))
                        fig_diff.tight_layout()
                        st.pyplot(fig_diff)

                else:
                    st.warning(f"El CSV debe contener al menos las columnas: {', '.join(required_cols)}")
            except Exception as e:
                st.error(f"Error al procesar el archivo CSV: {e}")

    # ---------------- TAB 5: Check URL ----------------
    with tab5:
        st.subheader("🔎 Detección de Phishing en URLs")

        # Entrada de la URL
        input_url = st.text_input("Introduce la URL:", placeholder="https://ejemplo.com")

        if st.button("🔍 Analizar URL"):
            if input_url:
                detected_url = detect_protocol(input_url)  # 🔍 Detectamos HTTP o HTTPS

                if detected_url:
                    st.info(f"🔗 Usando la URL detectada: {detected_url}")

                    # Mostrar spinner de carga mientras se procesa la URL
                    with st.spinner("📡 Analizando la URL, esto puede tardar unos segundos..."):
                        try:
                            # Cargar el dataset de entrenamiento
                            train_df = pd.read_parquet("Data/train.parquet")

                            # Extraer variables de la URL
                            df_url = extract_variables_from_url(detected_url)
                            df_url = pd.DataFrame([df_url])  # Convertir diccionario en DataFrame
                            df_url = df_url.apply(pd.to_numeric, errors="ignore")  # Convertir números correctamente
                            df_url = df_url.astype({col: "int32" for col in df_url.select_dtypes(include=["int8"]).columns})

                            # Asegurar que las columnas estén en el mismo orden que en el modelo entrenado
                            df_url = df_url[[col for col in train_df.columns if col != "status"]]

                            # Comprobar si la URL devuelve un DataFrame vacío
                            if df_url.isnull().all().all():
                                st.warning("⚠️ La URL no existe o es inaccesible.")
                                st.stop()  # Detener ejecución

                            # Obtener el modelo y el pipeline
                            model = st.session_state.get("model", None)
                            pipeline = st.session_state.get("pipeline", None)

                            if pipeline is not None and model is not None:
                                try:
                                    # Transformar los datos con el pipeline
                                    df_processed = pipeline.transform(df_url)

                                    # Intentar obtener las columnas esperadas por el modelo
                                    try:
                                        expected_features = model.get_booster().feature_names
                                    except AttributeError:
                                        expected_features = df_processed.columns.tolist()  # Alternativa

                                    # Obtener columnas actuales en df_processed
                                    actual_features = df_processed.columns.tolist()

                                    # Detectar discrepancias
                                    missing_in_df = set(expected_features) - set(actual_features)
                                    extra_in_df = set(actual_features) - set(expected_features)

                                    # Si hay discrepancias, mostrarlas y detener ejecución
                                    if missing_in_df:
                                        st.error(f"🚨 FALTAN columnas en df_processed: {missing_in_df}")
                                        st.stop()

                                    if extra_in_df:
                                        st.warning(f"🚨 Columnas extra en df_processed (pueden ser irrelevantes pero deberían revisarse): {extra_in_df}")

                                    # **Verificación final del orden de las columnas**
                                    if expected_features != actual_features:
                                        st.error("🚨 El orden de las columnas NO coincide con lo esperado. Se recomienda revisar la transformación del pipeline.")
                                        st.stop()

                                    # Realizar la predicción
                                    if hasattr(model, "predict_proba"):
                                        proba = model.predict_proba(df_processed)[:, 1][0]
                                        es_phishing = proba > 0.5
                                        confianza = proba * 100 if es_phishing else (1 - proba) * 100

                                        # Mostrar resultado
                                        resultado_texto = "⚠️ **La URL es PHISHING.**" if es_phishing else "✅ **La URL es LEGÍTIMA.**"
                                        st.markdown(f"### {resultado_texto}")
                                        st.markdown(f"**Nivel de confianza:** {confianza:.2f}%")

                                        # Información adicional
                                        st.info(
                                            "📌 Esta es una predicción basada en modelos de machine learning. "
                                            "Para conocer la metodología y las métricas, revisa los análisis detallados en las otras pestañas."
                                        )
                                    else:
                                        st.error("❌ El modelo no soporta `predict_proba`.")
                                except Exception as e:
                                    st.error(f"❌ Error en pipeline.transform o predict_proba: {e}")
                            else:
                                st.error("❌ Modelo o pipeline no cargados. Por favor, cárgalos en la barra lateral.")

                        except Exception as e:
                            st.error(f"❌ Error al procesar la URL, inténtalo de nuevo más tarde: {e}")
                else:
                    st.error("❌ La URL no es accesible. Verifica que la escribiste correctamente o que la web sigue activa.")
            else:
                st.warning("⚠️ Por favor, introduce una URL válida.")


        # 🔍 Explicación sobre la predicción del tráfico web
        with st.expander("📊 ¿Cómo se estima el tráfico web?"):
            st.markdown(
                """
                - **El volumen de búsqueda es un valor estimado** usando un modelo de *stacking* basado en el ranking de **Tranco**.
                - El rendimiento del modelo en pruebas fue:
                    - **R² = 0.8551**
                    - **RMSE = 1.34**
                - Modelos utilizados y sus métricas de error:
                
                | **Modelo**               | **RMSE**  | **R²**    |
                |-------------------------|---------|---------|
                | Regresión Lineal        | 2.05    | 0.6616  |
                | Regresión Polinómica    | 2.05    | 0.6616  |
                | Random Forest           | 1.37    | 0.8495  |
                | XGBoost                 | 1.58    | 0.7995  |
                | Stacking                | **1.34** | **0.8551**  |
                
                📌 Puedes ver más detalles en el notebook `webtraffic_modelling.ipynb` en el repositorio:
                🔗 [GitHub - Detector_Phishing](https://github.com/jarkillo/Detector_Phishing)
                """
            )

        # 🔍 FAQ y Preguntas Frecuentes
        with st.expander("❓ ¿Cómo funciona el modelo de detección de phishing?"):
            st.markdown(
                """
                Nuestro modelo de machine learning analiza diversas características de la URL, incluyendo:
                - **Estructura de la URL** (longitud, caracteres sospechosos, subdominios, etc.)
                - **Contenido de la página** (formularios, redirecciones, enlaces sospechosos, etc.)
                - **Información del dominio** (edad del dominio, registro WHOIS, DNS, etc.)
                - **Trafico web estimado** (basado en ranking de Tranco)

                Luego, usamos un **modelo predictivo entrenado con datos de phishing y sitios legítimos** para estimar la probabilidad de que la URL sea maliciosa.
                """
            )

        with st.expander("❓ ¿Qué significa la probabilidad mostrada?"):
            st.markdown(
                """
                - Si la probabilidad es mayor al **50%**, consideramos que la URL **puede ser phishing**.
                - Si la probabilidad es menor al **50%**, la URL **parece legítima**.
                - El número mostrado indica la **confianza del modelo** en la predicción.
                
                📌 **IMPORTANTE**: Este es un sistema de predicción basado en datos previos. Siempre verifica manualmente antes de ingresar información sensible en una página.
                """
            )

        with st.expander("❓ ¿Cómo puedo saber si una página es segura?"):
            st.markdown(
                """
                Aquí tienes algunas recomendaciones básicas:
                ✅ **Revisa la URL:** Evita sitios con caracteres raros o dominios extraños.  
                ✅ **Comprueba el certificado SSL:** Asegúrate de que usa **HTTPS**.  
                ✅ **No ingreses datos personales sin verificar:** Especialmente en emails sospechosos.  
                ✅ **Desconfía de enlaces acortados:** Pueden ocultar la dirección real.  
                ✅ **Busca señales de autenticidad:** Logos, contacto real, políticas de privacidad.  
                
                Si tienes dudas, **usa nuestro detector de phishing antes de hacer clic en enlaces desconocidos**.
                """
            )

        with st.expander("❓ ¿Este sistema detecta el 100% de los ataques de phishing?"):
            st.markdown(
                """
                ❌ No, ningún sistema es 100% perfecto.  
                🚀 Sin embargo, este modelo **ha sido entrenado con miles de ejemplos reales** y tiene una alta precisión.  
                📌 Aún así, **recomendamos siempre verificar manualmente** antes de ingresar información sensible.  
                """
            )

        with st.expander("❓ ¿Dónde puedo reportar una URL sospechosa?"):
            st.markdown(
                """
                Si encuentras una URL sospechosa, puedes reportarla en:
                - **Google Safe Browsing:** [https://safebrowsing.google.com/safebrowsing/report_phish/](https://safebrowsing.google.com/safebrowsing/report_phish/)
                - **PhishTank:** [https://www.phishtank.com/](https://www.phishtank.com/)
                - **Microsoft Defender SmartScreen:** [https://www.microsoft.com/en-us/wdsi/support/report-unsafe-site](https://www.microsoft.com/en-us/wdsi/support/report-unsafe-site)
                """
            )


if __name__ == "__main__":
    main()
