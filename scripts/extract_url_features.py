import numpy as np
from scripts.feature_extractor import (
    is_URL_accessible,
    get_domain,
    extract_features
)
import requests
import dotenv
import os
import json
import scripts.content_features
import scripts.external_features
import pandas as pd  # Asegurar que está importado


# Definir DTYPE_OPTIMIZADO
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
    'http_in_path': 'int8',
    'https_token': 'int8',
    'char_repeat': 'int16',
    'sfh': 'int8',
    'url': 'object'
}

# Definir headers
headers = [
    "url", "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens",
    "nb_at", "nb_qm", "nb_and", "nb_or", "nb_eq", "nb_underscore", "nb_tilde",
    "nb_percent", "nb_slash", "nb_star", "nb_colon", "nb_comma", "nb_semicolumn",
    "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash", "http_in_path",
    "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port",
    "tld_in_path", "tld_in_subdomain", "abnormal_subdomain", "nb_subdomains",
    "prefix_suffix", "random_domain", "shortening_service", "path_extension",
    "nb_redirection", "nb_external_redirection", "length_words_raw", "char_repeat",
    "shortest_words_raw", "shortest_word_host", "shortest_word_path", "longest_words_raw",
    "longest_word_host", "longest_word_path", "avg_words_raw", "avg_word_host",
    "avg_word_path", "phish_hints", "domain_in_brand", "brand_in_subdomain",
    "brand_in_path", "suspecious_tld", "statistical_report", "nb_hyperlinks",
    "ratio_intHyperlinks", "ratio_extHyperlinks", "ratio_nullHyperlinks", "nb_extCSS",
    "ratio_intRedirection", "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors",
    "login_form", "external_favicon", "links_in_tags", "submit_email", "ratio_intMedia",
    "ratio_extMedia", "sfh", "iframe", "popup_window", "safe_anchor", "onmouseover",
    "right_clic", "empty_title", "domain_in_title", "domain_with_copyright",
    "whois_registered_domain", "domain_registration_length", "domain_age",
    "web_traffic", "dns_record", "google_index", "page_rank"
]

from bs4 import BeautifulSoup

# Función principal
# Función principal corregida
def extract_variables_from_url(url: str, dtype_optimized=DTYPE_OPTIMIZADO):
    # Inicializar el diccionario a partir de dtype_optimized
    variables = {var: None for var in dtype_optimized.keys()}

    # Verificar accesibilidad
    state, processed_url, page = is_URL_accessible(url)
    if not state:
        print(f"URL inaccesible: {url}")
        return variables

    try:
        content = page.content
        dom = BeautifulSoup(content, 'html.parser')
        features_df = extract_features(processed_url, status=None)
        features_dict = features_df.iloc[0].to_dict()

        for feature_name in headers:
            if feature_name in features_dict:
                variables[feature_name] = features_dict[feature_name]
        
        # Aseguramos que todas las claves de headers estén presentes en el diccionario
        for key in headers:
            if key not in variables:
                variables[key] = None

    except Exception as e:
        print(f"Error al procesar URL: {e}")
        return variables

    for var, dtype in dtype_optimized.items():
        try:
            variables[var] = np.array([variables[var]]).astype(dtype)[0]
        except (ValueError, TypeError):
            variables[var] = None

    return variables

    