import requests
import re


#################################################################################################################################
#               Number of hyperlinks present in a website (Kumar Jain'18)
#################################################################################################################################

def nb_hyperlinks(Href, Link, Media, Form, CSS, Favicon):
    return len(Href['internals']) + len(Href['externals']) +\
           len(Link['internals']) + len(Link['externals']) +\
           len(Media['internals']) + len(Media['externals']) +\
           len(Form['internals']) + len(Form['externals']) +\
           len(CSS['internals']) + len(CSS['externals']) +\
           len(Favicon['internals']) + len(Favicon['externals'])

# def nb_hyperlinks(dom):
#    """
#    Cuenta los enlaces (href) y fuentes (src) en el DOM.
#    Si no se encuentran elementos, se devuelve 0.
#    """
#    hrefs = dom.find_all("a", href=True) if dom else []
#    srcs = dom.find_all("img", src=True) if dom else []
#    return len(hrefs) + len(srcs)

#################################################################################################################################
#               Internal hyperlinks ratio (Kumar Jain'18)
#################################################################################################################################


def h_total(Href, Link, Media, Form, CSS, Favicon):
    return nb_hyperlinks(Href, Link, Media, Form, CSS, Favicon)

def h_internal(Href, Link, Media, Form, CSS, Favicon):
    return len(Href['internals']) + len(Link['internals']) + len(Media['internals']) +\
           len(Form['internals']) + len(CSS['internals']) + len(Favicon['internals'])


def internal_hyperlinks(Href, Link, Media, Form, CSS, Favicon):
    total = h_total(Href, Link, Media, Form, CSS, Favicon)
    if total == 0:
        return 0
    else :
        return h_internal(Href, Link, Media, Form, CSS, Favicon)/total


#################################################################################################################################
#               External hyperlinks ratio (Kumar Jain'18)
#################################################################################################################################


def h_external(Href, Link, Media, Form, CSS, Favicon):
    return len(Href['externals']) + len(Link['externals']) + len(Media['externals']) +\
           len(Form['externals']) + len(CSS['externals']) + len(Favicon['externals'])
           
           
def external_hyperlinks(Href, Link, Media, Form, CSS, Favicon):
    total = h_total(Href, Link, Media, Form, CSS, Favicon)
    if total == 0:
        return 0
    else :
        return h_external(Href, Link, Media, Form, CSS, Favicon)/total


#################################################################################################################################
#               Number of null hyperlinks (Kumar Jain'18)
#################################################################################################################################

def h_null(hostname, Href, Link, Media, Form, CSS, Favicon):
    return len(Href['null']) + len(Link['null']) + len(Media['null']) + len(Form['null']) + len(CSS['null']) + len(Favicon['null'])

def null_hyperlinks(hostname, Href, Link, Media, Form, CSS, Favicon):
    total = h_total(Href, Link, Media, Form, CSS, Favicon)
    if total==0:
        return 0
    return h_null(hostname, Href, Link, Media, Form, CSS, Favicon)/total

#################################################################################################################################
#               Extrenal CSS (Kumar Jain'18)
#################################################################################################################################


def external_css(CSS):
    return len(CSS['externals'])
    

#################################################################################################################################
#               Internal redirections (Kumar Jain'18)
#################################################################################################################################


def h_i_redirect(Href, Link, Media, Form, CSS, Favicon):
    count = 0
    for link in Href['internals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    for link in Link['internals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    for link in Media['internals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    for link in Form['internals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    for link in CSS['internals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    for link in Favicon['internals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    return count

def internal_redirection(Href, Link, Media, Form, CSS, Favicon):
    internals = h_internal(Href, Link, Media, Form, CSS, Favicon)
    if (internals>0):
        return h_i_redirect(Href, Link, Media, Form, CSS, Favicon)/internals
    return 0

#################################################################################################################################
#               External redirections (Kumar Jain'18)
#################################################################################################################################


def h_e_redirect(Href, Link, Media, Form, CSS, Favicon):
    count = 0
    for link in Href['externals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    for link in Link['externals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    for link in Media['externals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue
    for link in Media['externals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue 
    for link in Form['externals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue    
    for link in CSS['externals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue    
    for link in Favicon['externals']:
        try:
            r = requests.get(link)
            if len(r.history) > 0:
                count+=1
        except:
            continue    
    return count

def external_redirection(Href, Link, Media, Form, CSS, Favicon):
    externals = h_external(Href, Link, Media, Form, CSS, Favicon)
    if (externals>0):
        return h_e_redirect(Href, Link, Media, Form, CSS, Favicon)/externals
    return 0



#################################################################################################################################
#               Generates internal errors (Kumar Jain'18)
#################################################################################################################################

def h_i_error(Href, Link, Media, Form, CSS, Favicon):
    count = 0
    for link in Href['internals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in Link['internals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in Media['internals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in Form['internals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in CSS['internals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue  
    for link in Favicon['internals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    return count

def internal_errors(Href, Link, Media, Form, CSS, Favicon):
    internals = h_internal(Href, Link, Media, Form, CSS, Favicon)
    if (internals>0):
        return h_i_error(Href, Link, Media, Form, CSS, Favicon)/internals
    return 0

#################################################################################################################################
#               Generates external errors (Kumar Jain'18)
#################################################################################################################################


def h_e_error(Href, Link, Media, Form, CSS, Favicon):
    count = 0
    for link in Href['externals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in Link['externals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in Media['externals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in Form['externals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in CSS['externals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    for link in Favicon['externals']:
        try:
            if requests.get(link).status_code >=400:
                count+=1
        except:
            continue
    return count


def external_errors(Href, Link, Media, Form, CSS, Favicon):
    externals = h_external(Href, Link, Media, Form, CSS, Favicon)
    if (externals>0):
        return h_e_error(Href, Link, Media, Form, CSS, Favicon)/externals
    return 0


#################################################################################################################################
#               Having login form link (Kumar Jain'18)
#################################################################################################################################

def login_form(Form):
    if not isinstance(Form, dict) or not all(k in Form for k in ["internals", "externals", "null"]):
        return 0  # Si Form no es un diccionario válido, asumimos que no hay login form.

    p = re.compile(r'([a-zA-Z0-9\_])+.php')
    
    if len(Form['externals']) > 0 or len(Form['null']) > 0:
        return 1
    
    for form in Form.get('internals', []) + Form.get('externals', []):
        if p.match(form):
            return 1
            
    return 0


#################################################################################################################################
#               Having external favicon (Kumar Jain'18)
#################################################################################################################################

def external_favicon(Favicon):
    if isinstance(Favicon, dict) and "externals" in Favicon:
        return int(len(Favicon["externals"]) > 0)
    return 0



#################################################################################################################################
#               Submitting to email 
#################################################################################################################################

def submitting_to_email(Form):
    if not isinstance(Form, dict):
        return 0
    
    for form in Form.get("internals", []) + Form.get("externals", []):
        if "mailto:" in form or "mail()" in form:
            return 1
    return 0



#################################################################################################################################
#               Percentile of internal media <= 61 : Request URL in Zaini'2019 
#################################################################################################################################

def internal_media(Media):
    total = len(Media['internals']) + len(Media['externals'])
    internals = len(Media['internals'])
    try:
        percentile = internals / float(total) * 100
    except:
        return 0
    
    return percentile

#################################################################################################################################
#               Percentile of external media : Request URL in Zaini'2019 
#################################################################################################################################

def external_media(Media):
    total = len(Media['internals']) + len(Media['externals'])
    externals = len(Media['externals'])
    try:
        percentile = externals / float(total) * 100
    except:
        return 0
    
    return percentile

#################################################################################################################################
#               Check for empty title 
#################################################################################################################################

def empty_title(Title):
    return int(not bool(Title))


#################################################################################################################################
#               Percentile of safe anchor : URL_of_Anchor in Zaini'2019 (Kumar Jain'18)
#################################################################################################################################

def safe_anchor(Anchor):
    if isinstance(Anchor, dict) and "safe" in Anchor and "unsafe" in Anchor:
        total = len(Anchor["safe"]) + len(Anchor["unsafe"])
        if total == 0:
            return 0
        return (len(Anchor["unsafe"]) / total) * 100
    return 0  # Si Anchor no es un diccionario válido, devolvemos 0


#################################################################################################################################
#               Percentile of internal links : links_in_tags in Zaini'2019 but without <Meta> tag
#################################################################################################################################

def links_in_tags(Link):
    total = len(Link['internals']) +  len(Link['externals'])
    internals = len(Link['internals'])
    try:
        percentile = internals / float(total) * 100
    except:
        return 0
    return percentile

#################################################################################################################################
#              Server Form Handler
#################################################################################################################################

def sfh(Form):
    """
    Determina si existe alguna entrada en la clave 'null' del diccionario Form.
    Devuelve 1 si existe al menos un valor en Form['null'], y 0 en caso contrario.
    Si Form no es un diccionario o no tiene la clave 'null', se asume que la estructura es inválida y devuelve 0.
    """
    if not isinstance(Form, dict):
        print(f"⚠️ sfh esperaba un diccionario, pero recibió: {Form} (Tipo: {type(Form)})")
        return 0
    if "null" not in Form:
        print(f"⚠️ sfh: La clave 'null' no está presente en Form: {Form}")
        return 0
    # Verificamos que Form['null'] sea una lista; si no, intentamos convertirlo a lista
    null_value = Form.get('null')
    if not isinstance(null_value, list):
        print(f"⚠️ sfh esperaba que Form['null'] sea una lista, pero es: {null_value} (Tipo: {type(null_value)})")
        return 0
    return 1 if len(null_value) > 0 else 0



#################################################################################################################################
#              IFrame Redirection
#################################################################################################################################

def iframe(IFrame):
    if isinstance(IFrame, dict) and "invisible" in IFrame:
        return int(len(IFrame["invisible"]) > 0)
    return 0
 
#################################################################################################################################
#              Onmouse action
#################################################################################################################################

def onmouseover(content):
    content_str = str(content).lower().replace(" ", "")
    return int('onmouseover="window.status=' in content_str)


#################################################################################################################################
#              Pop up window
#################################################################################################################################

def popup_window(content):
    content_str = str(content).lower()
    return int("prompt(" in content_str)


#################################################################################################################################
#              Right_clic action
#################################################################################################################################

def right_clic(content):
    if isinstance(content, str) and re.findall(r"event.button ?== ?2", content):
        return 1
    return 0


#################################################################################################################################
#              Domain in page title (Shirazi'18)
#################################################################################################################################

def domain_in_title(domain, title):
    if not title:  # Si el título es None o vacío, se asume que el dominio no está
        return 1
    return int(domain.lower() not in title.lower())


#################################################################################################################################
#              Domain after copyright logo (Shirazi'18)
#################################################################################################################################

def domain_with_copyright(domain, content):
    if not content:
        return 0  # Si no hay contenido, asumimos que no hay copyright

    try:
        match = re.search(r'(\u00A9|\u2122|\u00AE)', content)  # COPYRIGHT, TRADEMARK, REGISTERED
        if match:
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            _copyright = content[start:end]
            return int(domain.lower() not in _copyright.lower())
    except:
        pass  # Si hay algún error, asumimos que no hay copyright

    return 0

