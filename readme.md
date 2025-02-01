# Phishing URL Detection System

[Leer Readme en Español](#espanol)

## DEMO

https://detectorphishing.streamlit.app/

## Required .env for Local Execution or secrets.toml in a .streamlit Folder with the Following Data:

```
OPR_API_KEY = "<Open Page Rank API key>"
WHOIS_API_KEY = "<whoisxmlapi API key> #(if you want to use the paid version)"
SERP_API_KEY = "<SerpAPI key>"
CACHE_EXPIRATION_DAYS = "Number of days before the serpapi search cache expires"
SEARCH_HISTORY_FILE = "Location of the search.history.json file"  # This file includes searches performed by the program to avoid incurring extra serpAPI costs
```

## Paid WHOIS API

If you wish to use the paid API for the domain age from WHOIS, you need to uncomment the code in `external_features.py`.  
In principle, the whois library works well.

## Description

This project implements a phishing URL detection system using an interactive application developed with Streamlit. The system allows you to load pre-trained models, analyze datasets, generate predictions, and visualize performance metrics. Additionally, it offers advanced functionalities for detailed error analysis and comparisons between different models.

## Features

- **Model and Pipeline Loading:** Imports pre-trained models and pipelines to perform predictions.
- **Data Visualization:** Displays preliminary views of the loaded datasets.
- **Prediction Generation:** Makes predictions on URLs and shows basic metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC.
- **Advanced Analysis:** Provides detailed analysis of errors and model strengths, including feature importance graphs.
- **Model Comparison:** Enables the comparison of multiple models by loading a CSV file with specific metrics.
- **Result Download:** Facilitates the download of complete predictions and error details in CSV format.

## Project Structure

```
# Project Structure

## 📂 Data
- `allbrands.txt`
- `comparativa_modelos.csv`
- `test_dataset_F.csv`
- `test.parquet`
- `train_with_estimated_web_traffic.csv`
- `train_with_predicted_traffic.csv`
- `train_with_tranco.csv`
- `train.csv`
- `train.parquet`
- `tranco_list.csv`

## 📂 Models
- `__init__.py`
- `mejor_modelo.pkl`
- `mejor_pipeline.pkl`
- `metadatos.pkl`
- `web_traffic_stacking.pkl`

## 📂 Notebooks
- `Informe Final.docx`
- `Informe Final.pdf`
- `Phishing_modelling.ipynb`
- `Phising_EDA.ipynb`
- `webtraffic_modelling.ipynb`

## 📂 Results
_(Folder is empty or files are not listed)_

## 📂 scripts
- `__init__.py`
- `content_features.py`
- `external_features.py`
- `extract_url_features.py`
- `feature_extractor.py`
- `pandas2arff.py`
- `url_features.py`

## 📂 tests
- `__init__.py`
- `check_variables.py`
- `test_extract_features.py`
- `test_results.json`

## 📂 tools
- `convert_to_parquet.py`
- `tranco_dataset_building.py`

## 📂 utils
- `__init__.py`
- `functions.py`
- `transformers.py`

## Root Files
- `.env`
- `.gitignore`
- `app.py`
- `readme.md`
- `requirements.txt`
- `search_history.json`
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/phishing-detection-system.git
cd phishing-detection-system
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

If you are going to use new datasets, before running the application, it is necessary to convert the CSV files to Parquet format to optimize performance.

### 1. Run the Conversion Script (Not necessary to run, the Parquet files are already in the Data folder)

```bash
python scripts/convert_to_parquet.py
```

This script will load `train.csv` and `test_dataset_F.csv` from the `Data` folder, process them, and save them as `train.parquet` and `test.parquet` respectively in the same directory.

## EDA (No Need to Run)

The notebook `Phising_EDA.ipynb` contains the complete process of dataset exploration, including feature creation.

## Model Training (No Need to Run the Notebook)

The notebook `Phishing_modelling.ipynb` contains the complete process of model training, including feature selection, hyperparameter tuning, and evaluation. Once trained, the model, pipeline, and metadata are saved in the `Models` folder.

## Using the Streamlit Application

### 1. Run the Application

```bash
streamlit run app.py
```

### 2. User Interface

- **Sidebar:**
  - **Load Model and Pipeline:** Button to load the files `mejor_modelo.pkl`, `mejor_pipeline.pkl`, and `metadatos.pkl`.
  - **Load Dataset:** Select and load one of the available datasets (`train.parquet` or `test.parquet`).
  - **Decision Threshold:** Adjust the classification threshold for predictions.

- **Main Tabs:**
  - **📄 Data View:** Displays a preliminary view of the loaded dataset.
  - **📊 Prediction & Metrics:** Generates predictions and shows basic metrics.
  - **📈 Advanced Analysis:** Provides detailed analysis of errors and model strengths.
  - **🆚 Model Comparison:** Allows comparing different models by loading a CSV file with metrics.

### 3. Downloading Results

Within the Prediction and Advanced Analysis tabs, you will find options to download the complete predictions and errors in CSV format.

## Model Comparison

To compare different models, upload a CSV file with the metrics of interest (for example, `comparativa_modelos.csv`) in the **🆚 Model Comparison** tab. The file must include at least the columns `Modelo`, `Prueba`, and `F1-Validation`. Comparative graphs will be generated automatically.

## Contributions

Contributions are welcome. Please open an issue or submit a pull request with your improvements or fixes.

## Contact

For any questions or suggestions, please contact [manuellopezonline@gmail.com](mailto:manuellopezonline@gmail.com).


<a id="espanol"></a>

# Sistema de Detección de URLs Phishing

## DEMO

https://detectorphishing.streamlit.app/

## .env requerido para ejecutar en local o archivo secrets.toml en una carpeta .streamlit con los siguientes datos:

OPR_API_KEY = "<clave API de Open Page Rank>"
WHOIS_API_KEY = "<clave API de whoisxmlapi> #(si quieres usar la version de pago)"
SERP_API_KEY = "<Clave API de SerpAPI>"
CACHE_EXPIRATION_DAYS = "Dias para expirar la cache de busqueda de serpapi"
SEARCH_HISTORY_FILE = "Localizacion del archivo search.history.json" #Esto incluye las busquedas realizadas con el programa, para evitar sobrecoste de serpAPI

## API whois de pago

Si deseas usar la API de pago para el domain age del whois, necesitas descomentar el codigo en external_features.py
En principio la libreria de whois funciona bien

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
# Estructura del Proyecto

## 📂 Data
- `allbrands.txt`
- `comparativa_modelos.csv`
- `test_dataset_F.csv`
- `test.parquet`
- `train_with_estimated_web_traffic.csv`
- `train_with_predicted_traffic.csv`
- `train_with_tranco.csv`
- `train.csv`
- `train.parquet`
- `tranco_list.csv`

## 📂 Modelos
- `__init__.py`
- `mejor_modelo.pkl`
- `mejor_pipeline.pkl`
- `metadatos.pkl`
- `web_traffic_stacking.pkl`

## 📂 Notebooks
- `Informe Final.docx`
- `Informe Final.pdf`
- `Phishing_modelling.ipynb`
- `Phising_EDA.ipynb`
- `webtraffic_modelling.ipynb`

## 📂 Results
_(Carpeta vacía o sin listar archivos)_

## 📂 scripts
- `__init__.py`
- `content_features.py`
- `external_features.py`
- `extract_url_features.py`
- `feature_extractor.py`
- `pandas2arff.py`
- `url_features.py`

## 📂 tests
- `__init__.py`
- `check_variables.py`
- `test_extract_features.py`
- `test_results.json`

## 📂 tools
- `convert_to_parquet.py`
- `tranco_dataset_building.py`

## 📂 utils
- `__init__.py`
- `functions.py`
- `transformers.py`

## Archivos en la raíz
- `.env`
- `.gitignore`
- `app.py`
- `readme.md`
- `requirements.txt`
- `search_history.json`
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
