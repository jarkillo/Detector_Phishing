import requests

url = "https://app.novatalent.com"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive"
}

try:
    response = requests.get(url, headers=headers, timeout=5)
    print(f"🔍 Código de respuesta: {response.status_code}")
    print(f"🔄 URL final después de redirecciones: {response.url}")
    print(f"🔍 Headers de la respuesta: {response.headers}")

    if response.status_code == 403:
        print("🚨 El sitio está bloqueando peticiones automáticas (Error 403)")
    elif response.status_code >= 400:
        print("⚠️ El sitio no está accesible (Error 4xx o 5xx)")
    else:
        print("✅ La URL es accesible")
except requests.RequestException as e:
    print(f"❌ Error en la petición: {e}")
