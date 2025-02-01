#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:58:48 2020

@author: hannousse
@Edited by: jarko
"""

from datetime import datetime
from bs4 import BeautifulSoup
import requests
import whois
import time
import re
from dotenv import load_dotenv
import os
import json
import pandas as pd
import streamlit as st

# Cargar las variables del .env en entorno local
load_dotenv()

def get_config(key):
    """
    Intenta obtener la variable 'key' desde st.secrets; si no se encuentra,
    la obtiene desde os.environ.
    """
    return st.secrets.get(key) or os.getenv(key)

# Ahora, obtener las variables de configuración
OPR_API_KEY = get_config('OPR_API_KEY')
WHOIS_API_KEY = get_config('WHOIS_API_KEY')
SERP_API_KEY = get_config('SERP_API_KEY')
CACHE_EXPIRATION_DAYS = get_config('CACHE_EXPIRATION_DAYS')
SEARCH_HISTORY_FILE = get_config('SEARCH_HISTORY_FILE')


#################################################################################################################################
#               Domain registration age (jarko)
#################################################################################################################################
import whois
from datetime import datetime

def domain_age(domain): # Version gratis
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date

        if isinstance(creation_date, list):  # Algunos WHOIS devuelven lista
            creation_date = creation_date[0]

        if creation_date:
            age_days = (datetime.now() - creation_date).days
            return age_days
        return -1  # Si no hay fecha de creación
    except Exception as e:
        print(f"❌ domain_age ERROR: {e}")
        return -1

''' Version de pago
def domain_age(domain):
    """
    Consulta la antigüedad de un dominio usando la API de WhoisXML.
    :param domain: Dominio a consultar.
    :param api_key: Clave API de WhoisXML.
    :return: Fecha de creación del dominio o None si falla.
    """
    API_KEY = WHOIS_API_KEY
    url = f"https://www.whoisxmlapi.com/whoisserver/WhoisService?apiKey={API_KEY}&domainName={domain}&outputFormat=JSON"
    try:
        # Prueba la función con un dominio de ejemplo
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
        data = response.json()
        creation_date = data.get("WhoisRecord", {}).get("registryData", {}).get("createdDate", None)
        return creation_date
    except Exception as e:
        print(f"Error al consultar la antigüedad del dominio: {e}")
        return None
'''

import whois
from datetime import datetime, timezone

def domain_registration_length(domain):
    try:
        res = whois.whois(domain)
        creation_date = res.creation_date

        print(f"🔍 WHOIS para {domain} → Creation Date: {creation_date}")

        if creation_date is None:
            print(f"⚠️ {domain} no tiene fecha de creación en WHOIS.")
            return 0

        if isinstance(creation_date, list):
            creation_date = [date.astimezone(timezone.utc) if date.tzinfo else date.replace(tzinfo=timezone.utc) for date in creation_date]
            creation_date = min(creation_date)  # Tomamos la más antigua
            print(f"📌 Múltiples fechas en WHOIS, usando la más temprana: {creation_date}")

        elif isinstance(creation_date, datetime):
            if creation_date.tzinfo is None:
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            else:
                creation_date = creation_date.astimezone(timezone.utc)

        else:
            print(f"⚠️ `creation_date` tiene un tipo inesperado: {type(creation_date)}")
            return -1

        today = datetime.now(timezone.utc)
        print(f"🕒 Fecha actual en UTC: {today}")

        days_since_creation = abs((today - creation_date).days)
        print(f"✅ Días desde la creación del dominio: {days_since_creation}")

        return days_since_creation

    except Exception as e:
        print(f"❌ Error en `domain_registration_length` para {domain}: {e}")
        return -1






#################################################################################################################################
#               Domain recognized by WHOIS
#################################################################################################################################

 
import whois

def whois_registered_domain(domain):
    try:
        whois_info = whois.whois(domain)
        hostname = whois_info.domain_name

        if not hostname:
            return 1  # No hay información WHOIS

        if isinstance(hostname, list):
            return int(all(re.search(host.lower(), domain) is None for host in hostname))
        else:
            return int(re.search(hostname.lower(), domain) is None)
    except:
        return 1  # Si falla la consulta WHOIS, asumimos que el dominio no está registrado


#################################################################################################################################
#               Predicted web_traffic
#################################################################################################################################

import requests
from urllib.parse import urlparse
import joblib
import numpy as np
import tldextract

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 📌 Carpeta raíz del proyecto
tranco_csv_path = os.path.join(BASE_DIR, "Data", "tranco_list.csv")
model_path = os.path.join(BASE_DIR, "Modelos", "web_traffic_stacking.pkl")


def get_domain(url):
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
    return domain

# 📌 2️⃣ Función para obtener `tranco_rank`
def get_tranco_rank(domain, tranco_csv_path=tranco_csv_path):
    """
    Busca el ranking de Tranco en un CSV previamente descargado.
    
    :param domain: Dominio de la URL.
    :param tranco_csv_path: Ruta del archivo CSV con el ranking de Tranco.
    :return: Ranking de Tranco (entero) o 1_500_000 si no se encuentra.
    """
    try:
        tranco_df = pd.read_csv(tranco_csv_path, header=None, names=["rank", "domain"])

        tranco_dict = dict(zip(tranco_df['domain'], tranco_df['rank']))  # Convertimos en diccionario rápido
        return tranco_dict.get(domain, 1_500_000)  # Si no está en Tranco, devolvemos 1_500_000
    except Exception as e:
        print(f"❌ Error al cargar Tranco: {e}")
        return 1_500_000


def web_traffic(url):
        """
        Predice el tráfico web estimado a partir del ranking de Tranco.
        
        :param tranco_rank: Ranking de Tranco del dominio.
        :return: Predicción de tráfico web.
        """

        domain = get_domain(url)  # Extraemos el dominio
        tranco_rank = get_tranco_rank(domain, tranco_csv_path)  # Buscamos en el ranking de Tranco

        # 📌 1. Aplicar transformación logarítmica
        log_tranco_rank = np.log1p(tranco_rank)

        # 📌 2. Cargar el modelo entrenado
        modelo = joblib.load(model_path)

        # 📌 3. Convertir en DataFrame para la predicción
        df_input = pd.DataFrame({"log_tranco_rank": [log_tranco_rank]})

        # 📌 4. Hacer la predicción
        predicted_traffic = modelo.predict(df_input)[0]

        return predicted_traffic

#################################################################################################################################
#               Domain age of a url
#################################################################################################################################


#################################################################################################################################
#               Global rank
#################################################################################################################################

def global_rank(domain):
    rank_checker_response = requests.post("https://www.checkpagerank.net/index.php", {
        "name": domain
    })
    
    try:
        return int(re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0])
    except:
        return -1


#################################################################################################################################
#               Google index
#################################################################################################################################
def normalize_domain(domain):
    """Elimina 'www.' para normalizar comparaciones"""
    return domain.lower().replace("www.", "", 1)

import requests
import os
from urllib.parse import urlencode, urlparse
import time

def load_search_history():
    """Carga el historial desde JSON o devuelve un diccionario vacío si no existe."""
    if not os.path.exists(SEARCH_HISTORY_FILE):
        return {}
    
    try:
        with open(SEARCH_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_search_history(history):
    """Guarda el historial en el JSON."""
    with open(SEARCH_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

def fetch_search_archive(search_id):
    """ Recupera una búsqueda previa de SerpAPI si está en el historial. """
    archive_url = f"https://serpapi.com/searches/{search_id}.json?api_key={SERP_API_KEY}"
    try:
        response = requests.get(archive_url)
        if response.status_code == 200:
            return response.json()
        print(f"⚠️ No se pudo recuperar búsqueda {search_id}: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"❌ Error de red en `fetch_search_archive()`: {str(e)[:100]}...")
    return None

def google_index(url):
    """Consulta el índice de Google usando SerpAPI y evita búsquedas duplicadas."""
    search_history = load_search_history()

    # 📌 1️⃣ Intentar recuperar el `search_id` guardado
    if url in search_history:
        search_id = search_history[url].get("search_id")
        if search_id:
            api_url = f"https://serpapi.com/searches/{search_id}.json?api_key={SERP_API_KEY}"
            response = requests.get(api_url)

            if response.status_code == 200:
                print(f"✅ Usando búsqueda en caché para {url}")
                data = response.json()

                return process_serpapi_response(data)
            else:
                print(f"⚠️ No se pudo recuperar búsqueda `{search_id}` ({response.status_code}): {response.text}")
                del search_history[url]  # Eliminar caché corrupta
                save_search_history(search_history)  # Guardar cambios

    # 📌 2️⃣ Si no hay caché o falló, hacer una nueva búsqueda
    search_url = f"https://serpapi.com/search.json?q=site:{url}&api_key={SERP_API_KEY}"
    response = requests.get(search_url)

    if response.status_code == 200:
        data = response.json()
        search_history[url] = {"search_id": data.get("search_metadata", {}).get("id")}
        save_search_history(search_history)
        return process_serpapi_response(data)
    
    print(f"❌ No se pudo obtener datos de Google Index para {url}: {response.status_code}")
    return 0



def process_serpapi_response(data):
    try:
        total_results = data["search_information"].get("total_results", 0)
        return 0 if total_results > 0 else 1  # 0 si está indexado, 1 si no lo está
    except KeyError:
        return 1  # Si no existe la clave, asumimos que no está indexado




#################################################################################################################################
#               DNSRecord  expiration length
#################################################################################################################################

import dns.resolver

def dns_record(domain):
    try:
        answers = dns.resolver.resolve(domain, 'NS')
        return int(len(answers) == 0)
    except dns.resolver.NoAnswer:
        return 1  # No hay registros DNS
    except dns.resolver.NXDOMAIN:
        return 1  # Dominio no existe
    except dns.resolver.LifetimeTimeout:
        return 1  # Timeout en la consulta
    except:
        return 1  # Cualquier otro error

#################################################################################################################################
#               Page Rank from OPR
#################################################################################################################################

def page_rank(domain):
    url = f'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D={domain}'
    try:
        request = requests.get(url, headers={'API-OPR': OPR_API_KEY})
        result = request.json()

        # 📌 Verificar si la respuesta tiene la estructura correcta
        if "response" in result and result["response"]:
            return result["response"][0].get("page_rank_integer", None)  # Devolver PageRank si existe, sino None
        else:
            print(f"⚠️ Open PageRank no devolvió datos para {domain}")
            return None  # Retornar None si no hay datos

    except Exception as e:
        print(f"❌ Error en PageRank para {domain}: {e}")
        return None  # Si hay fallo, devolver None
