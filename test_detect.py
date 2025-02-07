from app import detect_protocol
url = "app.novatalent.com"
detected_url = detect_protocol(url)
print(f"Resultado detect_protocol: {detected_url}")