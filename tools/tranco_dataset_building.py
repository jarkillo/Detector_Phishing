import pandas as pd
import tldextract

# 📌 Rutas de archivos
TRAIN_DATASET_PATH = "../Data/train.csv"
TRANCO_RANK_PATH = "../Data/tranco_list.csv"

# 📌 Cargar dataset original
df = pd.read_csv(TRAIN_DATASET_PATH)

# 📌 Extraer solo el dominio raíz de cada URL
def extract_domain(url):
    if pd.isna(url) or not isinstance(url, str):
        return None
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"  # Devuelve "google.com" en "https://www.google.com/search"

df["domain"] = df["url"].apply(extract_domain)

# 📌 Cargar la lista de Tranco (que ya tiene dominios en su formato correcto)
tranco_df = pd.read_csv(TRANCO_RANK_PATH, names=["rank", "domain"])

# 📌 Unir los datasets por dominio
df = df.merge(tranco_df, on="domain", how="left")

# 📌 Rellenar valores faltantes con 1_500_000 (fuera del ranking) y convertir a int
df["tranco_rank"] = df["rank"].fillna(1_500_000).astype(int)

# 📌 Eliminar columna temporal "rank"
df.drop(columns=["rank"], inplace=True)

# 📌 Guardar el dataset con la nueva columna
OUTPUT_PATH = "../Data/train_with_tranco.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Dataset con Tranco Rank guardado en: {OUTPUT_PATH}")
