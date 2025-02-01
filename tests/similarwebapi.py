import requests

SIMILAR_WEB_API_KEY = "fa9647f622524c42972a44717a8f9c70"  # Sustituye con tu clave real
domain = "google.com"  # Prueba con Google, que seguro tiene datos

url = f"https://api.similarweb.com/v1/website/{domain}/total-traffic-and-engagement/visits?api_key={SIMILAR_WEB_API_KEY}&country=world"

response = requests.get(url)
print(f"Status Code: {response.status_code}")
print("Response JSON:", response.text)
