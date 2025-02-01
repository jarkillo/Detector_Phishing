import pandas as pd
from pathlib import Path

# Configuración base
SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent  # Subir un nivel desde scripts
DATA_DIR = BASE_DIR / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Crear carpeta si no existe
 
# Configuración de dtypes optimizados (global para reutilizar)
DTYPE_OPTIMIZADO = {
    'ip': 'int8',
    'phish_hints': 'int8',
    'google_index': 'int8',
    'punycode': 'int8',
    'port': 'int8',
    'tld_in_path': 'int8',
    'tld_in_subdomain': 'int8',
    'abnormal_subdomain': 'int8',
    'prefix_suffix': 'int8',
    'random_domain': 'int8',
    'shortening_service': 'int8',
    'path_extension': 'int8',
    'domain_in_brand': 'int8',
    'brand_in_subdomain': 'int8',
    'brand_in_path': 'int8',
    'suspecious_tld': 'int8',
    'statistical_report': 'int8',
    'login_form': 'int8',
    'external_favicon': 'int8',
    'submit_email': 'int8',
    'iframe': 'int8',
    'popup_window': 'int8',
    'onmouseover': 'int8',
    'right_clic': 'int8',
    'empty_title': 'int8',
    'domain_in_title': 'int8',
    'domain_with_copyright': 'int8',
    'whois_registered_domain': 'int8',
    'dns_record': 'int8',
    'nb_dots': 'int8',
    'nb_hyphens': 'int8',
    'nb_at': 'int8',
    'nb_qm': 'int8',
    'nb_and': 'int8',
    'nb_or': 'int8',
    'nb_eq': 'int8',
    'nb_underscore': 'int8',
    'nb_tilde': 'int8',
    'nb_percent': 'int8',
    'nb_slash': 'int8',
    'nb_star': 'int8',
    'nb_colon': 'int8',
    'nb_comma': 'int8',
    'nb_semicolumn': 'int8',
    'nb_dollar': 'int8',
    'nb_space': 'int8',
    'nb_www': 'int8',
    'nb_com': 'int8',
    'nb_dslash': 'int8',
    'nb_redirection': 'int16',
    'nb_external_redirection': 'int16',
    'nb_subdomains': 'int8',
    'links_in_tags': 'float32',
    'nb_extCSS': 'int16',
    'nb_hyperlinks': 'int16',
    'length_url': 'int16',
    'length_hostname': 'int16',
    'length_words_raw': 'int16',
    'shortest_words_raw': 'int16',
    'shortest_word_host': 'int16',
    'shortest_word_path': 'int16',
    'longest_words_raw': 'int16',
    'longest_word_host': 'int16',
    'longest_word_path': 'int16',
    'domain_registration_length': 'int16',
    'domain_age': 'float32',
    'ratio_digits_url': 'float32',
    'ratio_digits_host': 'float32',
    'safe_anchor': 'float32',
    'ratio_intHyperlinks': 'float32',
    'ratio_extHyperlinks': 'float32',
    'ratio_nullHyperlinks': 'float32',
    'ratio_intRedirection': 'float32',
    'ratio_extRedirection': 'float32',
    'ratio_intErrors': 'float32',
    'ratio_extErrors': 'float32',
    'ratio_intMedia': 'float32',
    'ratio_extMedia': 'float32',
    'avg_words_raw': 'float32',
    'avg_word_host': 'float32',
    'avg_word_path': 'float32',
    'web_traffic': 'float32',
    'page_rank': 'float16',
    'url': 'object',
    # La columna 'ID' será incluida solo en el dataset de test
    'ID': 'object',
    # 'status' solo se procesa en train
    'status': 'category'
}


def cargar_dataset_phishing_train():
    """
    Carga 'train.csv' desde la carpeta 'Data' fuera de 'scripts'.
    """
    DATA_PATH = DATA_DIR / "train.csv"

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"ERROR: Archivo de entrenamiento no encontrado.\n"
            f"Ruta esperada: {DATA_PATH.absolute()}"
        )

    try:
        # Cargar dataset
        df_train = pd.read_csv(
            DATA_PATH,
            encoding="utf-8-sig",
            dtype=DTYPE_OPTIMIZADO,
            na_values=["", " ", "NA", "N/A", "null", "NaN", "nan", "None"],
            on_bad_lines="warn",
        )

        # Validar y mapear 'status'
        if "status" in df_train.columns:
            df_train["status"] = df_train["status"].str.lower().str.strip()
            valores_esperados = ["phishing", "legitimate"]
            if not set(df_train["status"].unique()).issubset(valores_esperados):
                raise ValueError(
                    f"Valores inválidos en 'status': {set(df_train['status'].unique()) - set(valores_esperados)}"
                )
            df_train["status"] = df_train["status"].map({"phishing": 1, "legitimate": 0}).astype("int8")

    except Exception as e:
        raise RuntimeError(f"Error cargando el dataset de entrenamiento: {e}")

    return df_train


def cargar_dataset_phishing_test(file_path):
    """
    Carga el dataset de test desde el archivo especificado.
    """
    DATA_PATH = Path(file_path)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"ERROR: Archivo de test no encontrado en {DATA_PATH.absolute()}")

    try:
        # Cargar dataset
        df_test = pd.read_csv(
            DATA_PATH,
            sep=";",
            encoding="utf-8-sig",
            dtype={**DTYPE_OPTIMIZADO, 'status': 'object'},  # Ignorar 'status' en test
            na_values=["", " ", "NA", "N/A", "null", "NaN", "nan", "None"],
            on_bad_lines="warn",
        )

        # Verificar que contiene 'ID' y no tiene 'status'
        if "ID" not in df_test.columns:
            raise ValueError(f"El dataset de test debe contener la columna 'ID'.")
        if "status" in df_test.columns:
            print("Aviso: 'status' no debería estar en el dataset de test. Ignorando esta columna.")

    except Exception as e:
        raise RuntimeError(f"Error cargando el dataset de test: {e}")

    return df_test


def main():
    # Cargar y guardar el dataset de entrenamiento
    df_train = cargar_dataset_phishing_train()
    if df_train.empty:
        print("[ERROR] El dataset de entrenamiento está vacío.")
        return
    try:
        df_train.to_parquet(DATA_DIR / "train.parquet", index=False)
        print("[OK] train.parquet guardado correctamente.")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar train.parquet: {e}")

    # Cargar y guardar el dataset de test
    test_path = DATA_DIR / "test_dataset_F.csv"
    df_test = cargar_dataset_phishing_test(test_path)
    if df_test.empty:
        print("[ERROR] El dataset de test está vacío.")
        return
    try:
        df_test.to_parquet(DATA_DIR / "test.parquet", index=False)
        print("[OK] test.parquet guardado correctamente.")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar test.parquet: {e}")


if __name__ == "__main__":
    main()