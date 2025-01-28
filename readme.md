# Sistema de Detección de URLs Phishing

## Descripción

Este proyecto implementa un sistema de detección de URLs de phishing utilizando una aplicación interactiva desarrollada con Streamlit. El sistema permite cargar modelos preentrenados, analizar datasets, generar predicciones y visualizar métricas de rendimiento. Además, ofrece funcionalidades avanzadas para el análisis detallado de errores y comparativas entre diferentes modelos.

## Características

- **Carga de Modelos y Pipelines:** Importa modelos y pipelines preentrenados para realizar predicciones.
- **Visualización de Datos:** Muestra vistas preliminares de los datasets cargados.
- **Generación de Predicciones:** Realiza predicciones sobre URLs y muestra métricas básicas como Accuracy, Precisión, Recall, F1 Score y ROC AUC.
- **Análisis Avanzado:** Proporciona análisis detallados de errores y fortalezas del modelo, incluyendo gráficos de importancias de características.
- **Comparativa de Modelos:** Permite comparar múltiples modelos mediante la carga de un archivo CSV con métricas específicas.
- **Descarga de Resultados:** Facilita la descarga de predicciones completas y de errores en formato CSV.

## Estructura del Proyecto

```plaintext
.
├── app.py
├── requirements.txt
├── scripts
│   └── convert_to_parquet.py
├── Notebook
│   ├── Phising_EDA.ipynb
│   └── Phishing_modelling.ipynb
├── Modelos
│   ├── mejor_modelo.pkl
│   ├── mejor_pipeline.pkl
│   └── metadatos.pkl
├── Data
│   ├── comparativa_modelos.csv
│   ├── test_dataset_F.csv
│   ├── test.parquet
│   ├── train.csv
│   └── train.parquet
└── utils
    ├── functions.py
    └── transformers.py
```

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu_usuario/sistema-deteccion-phishing.git
cd sistema-deteccion-phishing
```

### 2. Crear un Entorno Virtual (Opcional pero Recomendado)

```bash
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
```

### 3. Instalar las Dependencias

```bash
pip install -r requirements.txt
```

## Preparación de Datos

Si vas a usar nuevos dataset, antes de ejecutar la aplicación, es necesario convertir los archivos CSV a formato Parquet para optimizar el rendimiento.

### 1. Ejecutar el Script de Conversión (No es necesario ejecutarlo, los parquet ya se encuentran en Data)

```bash
python scripts/convert_to_parquet.py
```

Este script cargará `train.csv` y `test_dataset_F.csv` desde la carpeta `Data`, los procesará y los guardará como `train.parquet` y `test.parquet` respectivamente en el mismo directorio.

## EDA (No es necesario ejecutarlo)

El notebook `Phising_EDA.ipynb` contiene el proceso completo de exploración del dataset, incluyendo la creación de características.

## Entrenamiento del Modelo (No es necesario ejecutar el Notebook)

El notebook `Phishing_modelling.ipynb` contiene el proceso completo de entrenamiento del modelo, incluyendo la selección de características, ajuste de hiperparámetros y evaluación. Una vez entrenado, el modelo, el pipeline y los metadatos se guardan en la carpeta `Modelos`.

## Uso de la Aplicación Streamlit

### 1. Ejecutar la Aplicación

```bash
streamlit run app.py
```

### 2. Interfaz de Usuario

- **Barra Lateral:**
  - **Cargar Modelo y Pipeline:** Botón para cargar los archivos `mejor_modelo.pkl`, `mejor_pipeline.pkl` y `metadatos.pkl`.
  - **Cargar Dataset:** Selecciona y carga uno de los datasets disponibles (`train.parquet` o `test.parquet`).
  - **Umbral de Decisión (Threshold):** Ajusta el umbral de clasificación para las predicciones.

- **Pestañas Principales:**
  - **📄 Vista de Datos:** Muestra una vista preliminar del dataset cargado.
  - **📊 Predicción & Métricas:** Genera predicciones y muestra métricas básicas.
  - **📈 Análisis Avanzado:** Proporciona análisis detallados de errores y fortalezas del modelo.
  - **🆚 Comparativa Modelos:** Permite comparar diferentes modelos mediante la carga de un archivo CSV con métricas.

### 3. Descargar Resultados

Dentro de las pestañas de Predicción y Análisis Avanzado, encontrarás opciones para descargar las predicciones completas y los errores en formato CSV.

## Comparativa de Modelos

Para comparar diferentes modelos, sube un archivo CSV con las métricas de interés (por ejemplo, `comparativa_modelos.csv`) en la pestaña **🆚 Comparativa Modelos**. El archivo debe contener al menos las columnas `Modelo`, `Prueba` y `F1-Validation`. Se generarán gráficos comparativos automáticamente.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request con tus mejoras o correcciones.

## Contacto

Para cualquier consulta o sugerencia, por favor contacta a [manuellopezonline@gmail.com](mailto:manuellopezonline@gmail.com).
