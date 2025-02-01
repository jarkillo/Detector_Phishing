import json
import pandas as pd
import sys
import os
import numpy as np

# Asegurar que la raíz del proyecto está en sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.extract_url_features import extract_variables_from_url

# 📌 URLs de prueba con distintos patrones
TEST_URLS = {
    "drive": "https://drive.google.com/file/d/1gyjW8k1wsZxk3eu4ylanEetEMSNN77m5/view?usp=sharing",
    "google": "https://www.google.com",
    
}

# 🛠 Almacenar resultados de prueba
results = {}

def convert_numpy_types(obj):
    """Convierte tipos de NumPy a tipos nativos de Python para JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# 🚀 Ejecutar pruebas
for label, url in TEST_URLS.items():
    print(f"🔍 Probando {label}: {url}")
    try:
        variables = extract_variables_from_url(url)
        
        if not isinstance(variables, dict):
            raise ValueError(f"La función extract_variables_from_url no devolvió un diccionario, sino {type(variables)}")

        results[label] = variables
    except Exception as e:
        print(f"❌ Error en {label}: {e}")
        results[label] = {"error": str(e)}

# 📊 Convertir resultados a DataFrame para análisis
df = pd.DataFrame.from_dict(results, orient="index")

# 💾 Guardar resultados en JSON para referencia
output_file = os.path.join(os.path.dirname(__file__), "test_results.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=6, default=convert_numpy_types)

# 📊 Mostrar análisis de las variables numéricas
print("\n📊 Resumen de resultados (variables numéricas):")
print(df.describe(include=[np.number]).T.sort_values(by="mean", ascending=False))

# 🔎 Revisar valores constantes o sospechosos
constant_values = df.nunique().sort_values()
print("\n🔍 Columnas con valores fijos (posibles errores):")
print(constant_values[constant_values == 1])

# 🔎 Revisar si hay valores inesperados en métricas clave
suspicious_columns = ["google_index", "page_rank", "web_traffic"]
for col in suspicious_columns:
    if col in df.columns:
        print(f"\n⚠️ {col} valores:")
        print(df[col].value_counts())

# 🔍 Revisar valores nulos

print("\n🔍 Variables no calculadas correctamente:")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"⚠️ {col}: {df[col].isnull().sum()} valores nulos")

print(f"\n✅ Test finalizado. Revisa `{output_file}` para más detalles.")

