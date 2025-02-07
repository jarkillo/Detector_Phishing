#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:23:31 2020

@author: hannousse
"""

import pandas as pd 
import urllib.parse
import tldextract
import requests
import json
import csv
import os
import re
from dotenv import load_dotenv
import os
import sys
from scripts.content_features import safe_request

# Asegurar que el directorio raíz está en sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importaciones corregidas
from scripts import content_features as ctnfe
from scripts import url_features as urlfe
from scripts import external_features as trdfe
from scripts.pandas2arff import pandas2arff
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import signal
import threading

import numpy as np
import tldextract
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import traceback  # Para debugging avanzados

class TimedOutExc(Exception):
    pass

def deadline(timeout):
    """
    Reemplaza la funcionalidad de `signal.SIGALRM` para sistemas que no lo soportan (Windows).
    """
    def decorate(func):
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                raise TimedOutExc(f"Timeout de {timeout} segundos alcanzado.")
            if exception[0]:
                raise exception[0]
            return result[0]

        return wrapper
    return decorate

@deadline(5)
def is_URL_accessible(url, timeout=10):
    """
    Verifica si una URL es accesible.
    
    Retorna una tupla (estado, url_procesada, objeto_response).
      - estado: True si se obtuvo respuesta y el código es menor a 400 (excepto 404), o False en caso contrario.
      - url_procesada: la URL con protocolo, o None si falla.
      - objeto_response: el objeto de respuesta de requests, o None en caso de error.
    
    Si se recibe un 404, se imprime un mensaje específico y se retorna False.
    """
    try:
        # Intentamos acceder a la URL
        response = safe_request(url, timeout=timeout)
        # Si se obtiene un 404, imprimimos un mensaje amigable y retornamos False
        if response.status_code == 404:
            print(f"❌ La URL {url} respondió con un 404 (página no encontrada).")
            return False, url, response
        # Si el código de estado es menor a 400, consideramos la URL accesible
        if response.status_code < 400 and response.content.strip() not in [b'', b' ']:
            return True, url, response
        else:
            print(f"❌ La URL {url} respondió con el código {response.status_code}.")
    except Exception as e:
        print(f"❌ Error al acceder a {url}: {e}")
    
    return False, None, None

def get_domain(url):
    o = urllib.parse.urlsplit(url)
    return o.hostname, tldextract.extract(url).domain, o.path




def getPageContent(url):
    parsed = urlparse(url)
    url = parsed.scheme+'://'+parsed.netloc
    try:
        page = safe_request(url)
    except:
        if not parsed.netloc.startswith('www'):
            url = parsed.scheme+'://www.'+parsed.netloc
            page = safe_request(url)
    if page.status_code != 200:
        return None, None
    else:    
        return url, page.content
 
    
    
#################################################################################################################################
#              Data Extraction Process
#################################################################################################################################

import requests
from bs4 import BeautifulSoup
import re

def extract_data_from_URL(hostname, content, domain, Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text):
    Null_format = ["", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever",
                   "#content", "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"]

    soup = BeautifulSoup(content, 'html.parser', from_encoding='iso-8859-1')

    # --- Collect all external and internal hrefs from URL ---
    for link in soup.find_all('a'):
        if not link.has_attr('href') or not link['href']:  # ✅ Evitar KeyError
            print(f"⚠️ Se encontró un `<a>` sin 'href': {link}")
            continue
        
        href = link['href']
        dots = [x.start(0) for x in re.finditer(r'\.', href)]

        if hostname in href or domain in href or len(dots) == 1 or not href.startswith('http'):
            if "#" in href or "javascript" in href.lower() or "mailto" in href.lower():
                Anchor['unsafe'].append(href)
            if not href.startswith('http'):
                if not href.startswith('/'):
                    Href['internals'].append(f"{hostname}/{href}")
                elif href in Null_format:
                    Href['null'].append(href)
                else:
                    Href['internals'].append(f"{hostname}{href}")
        else:
            Href['externals'].append(href)
            Anchor['safe'].append(href)

    # --- Collect all media sources ---
    for tag in soup.find_all(['img', 'audio', 'embed', 'iframe']):
        if not tag.has_attr('src') or not tag['src']:  # ✅ Evitar KeyError
            continue
        
        src = tag['src']
        dots = [x.start(0) for x in re.finditer(r'\.', src)]
        
        if hostname in src or domain in src or len(dots) == 1 or not src.startswith('http'):
            if not src.startswith('http'):
                if not src.startswith('/'):
                    Media['internals'].append(f"{hostname}/{src}")
                elif src in Null_format:
                    Media['null'].append(src)
                else:
                    Media['internals'].append(f"{hostname}{src}")
        else:
            Media['externals'].append(src)

    # --- Collect all link tags ---
    for link in soup.find_all('link'):
        if not link.has_attr('href') or not link['href']:  # ✅ Evitar KeyError
            continue
        
        href = link['href']
        dots = [x.start(0) for x in re.finditer(r'\.', href)]
        
        if hostname in href or domain in href or len(dots) == 1 or not href.startswith('http'):
            if not href.startswith('http'):
                if not href.startswith('/'):
                    Link['internals'].append(f"{hostname}/{href}")
                elif href in Null_format:
                    Link['null'].append(href)
                else:
                    Link['internals'].append(f"{hostname}{href}")
        else:
            Link['externals'].append(href)

    # --- Collect all CSS references ---
    for link in soup.find_all('link', rel='stylesheet'):
        if not link.has_attr('href') or not link['href']:  # ✅ Evitar KeyError
            continue
        
        href = link['href']
        dots = [x.start(0) for x in re.finditer(r'\.', href)]
        
        if hostname in href or domain in href or len(dots) == 1 or not href.startswith('http'):
            if not href.startswith('http'):
                if not href.startswith('/'):
                    CSS['internals'].append(f"{hostname}/{href}")
                elif href in Null_format:
                    CSS['null'].append(href)
                else:
                    CSS['internals'].append(f"{hostname}{href}")
        else:
            CSS['externals'].append(href)

    # --- Collect all form actions ---
    for form in soup.find_all('form'):
        if not form.has_attr('action') or not form['action']:  # ✅ Evitar KeyError
            continue
        
        action = form['action']
        dots = [x.start(0) for x in re.finditer(r'\.', action)]
        
        if hostname in action or domain in action or len(dots) == 1 or not action.startswith('http'):
            if not action.startswith('http'):
                if not action.startswith('/'):
                    Form['internals'].append(f"{hostname}/{action}")
                elif action in Null_format or action == 'about:blank':
                    Form['null'].append(action)
                else:
                    Form['internals'].append(f"{hostname}{action}")
        else:
            Form['externals'].append(action)

    # --- Collect all favicons ---
    for head in soup.find_all('head'):
        for link in head.find_all('link'):
            if not link.has_attr('href') or not link['href']:  # ✅ Evitar KeyError
                continue

            href = link['href']
            dots = [x.start(0) for x in re.finditer(r'\.', href)]

            if hostname in href or domain in href or len(dots) == 1 or not href.startswith('http'):
                if not href.startswith('http'):
                    if not href.startswith('/'):
                        Favicon['internals'].append(f"{hostname}/{href}")
                    elif href in Null_format:
                        Favicon['null'].append(href)
                    else:
                        Favicon['internals'].append(f"{hostname}{href}")
            else:
                Favicon['externals'].append(href)

    # --- Collect invisible iframes ---
    for i_frame in soup.find_all('iframe'):
        if not i_frame.has_attr('width') or not i_frame.has_attr('height') or not i_frame.has_attr('frameborder'):
            continue
        
        if i_frame['width'] == "0" and i_frame['height'] == "0" and i_frame['frameborder'] == "0":
            IFrame['invisible'].append(i_frame)
        else:
            IFrame['visible'].append(i_frame)

    # --- Get page title ---
    try:
        Title = soup.title.string if soup.title else ""
    except Exception as e:
        print(f"⚠️ Error al obtener el título: {e}")
        Title = ""

    # --- Get content text ---
    Text = soup.get_text()

    # --- Asegurar que Form es un diccionario bien estructurado ---
    if not isinstance(Form, dict):
        print(f"⚠️ Form no es un diccionario: {type(Form)} → {Form}")
        Form = {"internals": [], "externals": [], "null": []}

    if "null" not in Form:
        print("⚠️ La clave 'null' no existe en Form. Se inicializa vacía.")
        Form["null"] = []
    
    return Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text


#################################################################################################################################
#              Calculate features from extracted data
#################################################################################################################################

def extract_features(url, status):

    def words_raw_extraction(domain, subdomain, path):
        w_domain = re.split(r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", domain.lower())
        w_subdomain = re.split(r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", subdomain.lower())   
        w_path = re.split(r"\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", path.lower())
        raw_words = w_domain + w_path + w_subdomain
        w_host = w_domain + w_subdomain
        raw_words = list(filter(None, raw_words))
        return raw_words, list(filter(None, w_host)), list(filter(None, w_path))

    variables = []  # Aquí almacenamos los valores

    try:
        state, iurl, page = is_URL_accessible(url)
        if not state:
            print(f"❌ URL no accesible: {url}")
            return [None] * len(headers)

        content = page.content
        hostname, domain, path = get_domain(url)
        extracted_domain = tldextract.extract(url)
        domain = extracted_domain.domain + '.' + extracted_domain.suffix
        subdomain = extracted_domain.subdomain
        tmp = url[url.find(extracted_domain.suffix):len(url)]
        pth = tmp.partition("/")
        path = pth[1] + pth[2]
        words_raw, words_raw_host, words_raw_path = words_raw_extraction(extracted_domain.domain, subdomain, pth[2])
        tld = extracted_domain.suffix
        parsed = urlparse(url)
        scheme = parsed.scheme

        # 📌 🔥 Inicializar estructuras para `content_features.py`
        Href = {'internals': [], 'externals': [], 'null': []}
        Link = {'internals': [], 'externals': [], 'null': []}
        Anchor = {'safe': [], 'unsafe': [], 'null': []}
        Media = {'internals': [], 'externals': [], 'null': []}
        Form = {'internals': [], 'externals': [], 'null': []}
        CSS = {'internals': [], 'externals': [], 'null': []}
        Favicon = {'internals': [], 'externals': [], 'null': []}
        IFrame = {'visible': [], 'invisible': [], 'null': []}
        Title = ''
        Text = ''

        # Extraer datos del DOM con BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text = extract_data_from_URL(
            hostname, content, domain, Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text
        )

        # 🛠️ Si una función falla, simplemente devuelve `None` (sin valores inventados)
        #def safe_call(func, *args, default=-1):
         #   try:
          #      resultado = func(*args)
           #     print(f"✅ {func.__name__} → {resultado}")  # 🛠 DEBUG
            #    return resultado
            #except Exception as e:
             #   print(f"❌ {func.__name__} ERROR: {e}")  # ⚠️ Mostrar error
              #  return default

        # 📌 Generación de valores para TODAS las variables en headers
        values = {
            "url": urlfe.url_length(url),
            "length_url": urlfe.url_length(url),
            "length_hostname": urlfe.url_length(hostname),
            "ip": urlfe.having_ip_address(url),
            "nb_dots": urlfe.count_dots(url),
            "nb_hyphens": urlfe.count_hyphens(url),
            "nb_at": urlfe.count_at(url),
            "nb_qm": urlfe.count_exclamation(url),  # Asegurando que usamos el nombre correcto
            "nb_and": urlfe.count_and(url),
            "nb_or": urlfe.count_or(url),
            "nb_eq": urlfe.count_equal(url),
            "nb_underscore": urlfe.count_underscore(url),
            "nb_tilde": urlfe.count_tilde(url),
            "nb_percent": urlfe.count_percentage(url),
            "nb_slash": urlfe.count_slash(url),
            "nb_star": urlfe.count_star(url),
            "nb_colon": urlfe.count_colon(url),
            "nb_comma": urlfe.count_comma(url),
            "nb_semicolumn": urlfe.count_semicolumn(url),
            "nb_dollar": urlfe.count_dollar(url),
            "nb_space": urlfe.count_space(url),
            "nb_www": urlfe.check_www(words_raw),
            "nb_com": urlfe.check_com(words_raw),
            "nb_dslash": urlfe.count_double_slash(url),
            "http_in_path": urlfe.count_http_token(path),  # Corregido a http_in_path
            "https_token": urlfe.https_token(scheme),
            "ratio_digits_url": urlfe.ratio_digits(url),
            "ratio_digits_host": urlfe.ratio_digits(hostname),
            "punycode": urlfe.punycode(url),
            "port": urlfe.port(url),
            "tld_in_path": urlfe.tld_in_path(tld, path),
            "tld_in_subdomain": urlfe.tld_in_subdomain(tld, subdomain),
            "abnormal_subdomain": urlfe.abnormal_subdomain(url),
            "nb_subdomains": urlfe.count_subdomain(url),
            "prefix_suffix": urlfe.prefix_suffix(url),
            "random_domain": urlfe.random_domain(domain),
            "shortening_service": urlfe.shortening_service(url),
            "path_extension": urlfe.path_extension(path),
            "nb_redirection": ctnfe.nb_hyperlinks(Href, Link, Media, Form, CSS, Favicon),
            "nb_external_redirection": ctnfe.external_redirection(Href, Link, Media, Form, CSS, Favicon),

            # 🟢 CORRECCIÓN DE NOMBRES Y USO DE FUNCIONES EXISTENTES 🟢
            "length_words_raw": urlfe.length_word_raw(words_raw),  # ✅ CORRECTO
            "shortest_words_raw": urlfe.shortest_word_length(words_raw),  # ✅ CORRECTO
            "shortest_word_host": urlfe.shortest_word_length(words_raw_host),  # ✅ Aplicamos al host
            "shortest_word_path": urlfe.shortest_word_length(words_raw_path),  # ✅ Aplicamos al path
            "longest_words_raw": urlfe.longest_word_length(words_raw),  # ✅ CORRECTO
            "longest_word_host": urlfe.longest_word_length(words_raw_host),  # ✅ Aplicamos al host
            "longest_word_path": urlfe.longest_word_length(words_raw_path),  # ✅ Aplicamos al path
            "avg_words_raw": urlfe.average_word_length(words_raw),  # ✅ CORRECTO
            "avg_word_host": urlfe.average_word_length(words_raw_host),  # ✅ Aplicamos al host
            "avg_word_path": urlfe.average_word_length(words_raw_path),  # ✅ Aplicamos al path

            "phish_hints": urlfe.phish_hints(url),
            "domain_in_brand": urlfe.domain_in_brand(domain),
            "brand_in_subdomain": urlfe.tld_in_subdomain(domain, subdomain),
            "brand_in_path": urlfe.brand_in_path(domain, path),
            "suspecious_tld": urlfe.suspecious_tld(tld),
            "statistical_report": urlfe.statistical_report(url, domain),
            "nb_hyperlinks": ctnfe.nb_hyperlinks(Href, Link, Media, Form, CSS, Favicon),
            "ratio_intHyperlinks": ctnfe.internal_hyperlinks(Href, Link, Media, Form, CSS, Favicon),
            "ratio_extHyperlinks": ctnfe.external_hyperlinks(Href, Link, Media, Form, CSS, Favicon),
            "ratio_nullHyperlinks": ctnfe.null_hyperlinks(hostname, Href, Link, Media, Form, CSS, Favicon),
            "nb_extCSS": ctnfe.external_css(CSS),
            "ratio_intRedirection": ctnfe.internal_redirection(Href, Link, Media, Form, CSS, Favicon),
            "ratio_extRedirection": ctnfe.external_redirection(Href, Link, Media, Form, CSS, Favicon),
            "ratio_intErrors": ctnfe.internal_errors(Href, Link, Media, Form, CSS, Favicon),
            "ratio_extErrors": ctnfe.external_errors(Href, Link, Media, Form, CSS, Favicon),
            "login_form": ctnfe.login_form(Form),
            "external_favicon": ctnfe.external_favicon(Favicon),
            "links_in_tags": ctnfe.links_in_tags(Link),
            "submit_email": ctnfe.submitting_to_email(Form),
            "ratio_intMedia": ctnfe.internal_media(Media),
            "ratio_extMedia": ctnfe.external_media(Media),
            "sfh": ctnfe.sfh(Form),
            "iframe": ctnfe.iframe(url),
            "popup_window": ctnfe.popup_window(url),
            "safe_anchor": ctnfe.safe_anchor(url),
            "onmouseover": ctnfe.onmouseover(url),
            "right_clic": ctnfe.right_clic(url),
            "empty_title": ctnfe.empty_title(url),
            "domain_in_title": ctnfe.domain_in_title(domain, Title),
            "domain_with_copyright": ctnfe.domain_with_copyright(domain, Text),
            "whois_registered_domain": trdfe.whois_registered_domain(domain),
            "domain_registration_length": trdfe.domain_registration_length(domain),
            "domain_age": trdfe.domain_age(domain),
            "web_traffic": trdfe.web_traffic(url),
            "dns_record": trdfe.dns_record(domain),
            "google_index": trdfe.google_index(url),
            "page_rank": trdfe.page_rank(domain),
            "char_repeat": urlfe.char_repeat(words_raw),
        }
                    
        # Asegurar que la longitud final de variables sea 87
        if len(values) < len(headers):
            for key in headers:
                if key not in values:
                     values[key] = None


        print(f"📊 Longitud final de variables tras debug: {len(values)} / Esperado: {len(headers)}")

        import pandas as pd

        # Convertir el diccionario de valores en un DataFrame
        df = pd.DataFrame([values])

        # Asegurar que todas las columnas necesarias están presentes (y en orden)
        df = df.reindex(columns=headers)

        # Completar el diccionario con todas las claves de headers
        for key in headers:
            if key not in values:
                values[key] = None

        import pandas as pd
        df = pd.DataFrame([values])
        df = df.reindex(columns=headers)
        return df


    except Exception as e:
        print(f"❌ Error en `extract_features`: {e}")
        traceback.print_exc()
        return {header: None for header in headers}  # Devuelve un diccionario vacío con `None`

#################################################################################################################################
#             Intialization
#################################################################################################################################



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