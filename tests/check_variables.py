# Lista de variables en values (extraídas del código proporcionado)
values_list = [
    "url", "length_url", "length_hostname", "having_ip_address", "count_dots", "count_hyphens",
    "count_at", "count_exclamation", "count_and", "count_or", "count_equal", "count_underscore",
    "count_tilde", "count_percentage", "count_slash", "count_star", "count_colon", "count_comma",
    "count_semicolumn", "count_dollar", "count_space", "count_www", "count_com", "count_double_slash",
    "count_http_token", "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port",
    "tld_in_path", "tld_in_subdomain", "abnormal_subdomain", "count_subdomain", "prefix_suffix",
    "random_domain", "shortening_service", "path_extension", "google_index", "page_rank", "web_traffic",
    "dns_record", "domain_registration_length", "domain_age", "nb_hyperlinks", "internal_hyperlinks",
    "external_hyperlinks", "null_hyperlinks", "external_css", "internal_redirection",
    "external_redirection", "internal_errors", "external_errors", "login_form", "external_favicon",
    "links_in_tags", "submitting_to_email", "internal_media", "external_media", "domain_in_title",
    "domain_with_copyright"
]

# Lista de variables en headers (extraídas del código proporcionado)
headers_list = [
    "url", "length_url", "length_hostname", "having_ip_address", "count_dots", "count_hyphens",
    "count_at", "count_exclamation", "count_and", "count_or", "count_equal", "count_underscore",
    "count_tilde", "count_percentage", "count_slash", "count_star", "count_colon", "count_comma",
    "count_semicolumn", "count_dollar", "count_space", "count_www", "count_com", "count_double_slash",
    "count_http_token", "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port",
    "tld_in_path", "tld_in_subdomain", "abnormal_subdomain", "count_subdomains", "prefix_suffix",
    "random_domain", "shortening_service", "path_extension", "google_index", "page_rank", "web_traffic",
    "dns_record", "nb_hyperlinks", "internal_hyperlinks", "external_hyperlinks", "null_hyperlinks",
    "external_css", "internal_redirection", "external_redirection", "internal_errors", "external_errors",
    "login_form", "external_favicon", "links_in_tags", "submitting_to_email", "internal_media",
    "external_media", "domain_in_title", "domain_with_copyright", "length_words_raw",
    "shortest_word_raw", "shortest_word_host", "shortest_word_path", "longest_word_raw",
    "longest_word_host", "longest_word_path", "avg_word_raw", "avg_word_host", "avg_word_path",
    "phish_hints", "domain_registration_length", "domain_age", "suspecious_tld", "statistical_report"
]

real_columns = [
    "url","length_url","length_hostname","ip","nb_dots","nb_hyphens","nb_at","nb_qm","nb_and","nb_or","nb_eq",
    "nb_underscore","nb_tilde","nb_percent","nb_slash","nb_star","nb_colon","nb_comma","nb_semicolumn","nb_dollar",
    "nb_space","nb_www","nb_com","nb_dslash","http_in_path","https_token","ratio_digits_url","ratio_digits_host",
    "punycode","port","tld_in_path","tld_in_subdomain","abnormal_subdomain","nb_subdomains","prefix_suffix",
    "random_domain","shortening_service","path_extension","nb_redirection","nb_external_redirection",
    "length_words_raw","char_repeat","shortest_words_raw","shortest_word_host","shortest_word_path",
    "longest_words_raw","longest_word_host","longest_word_path","avg_words_raw","avg_word_host","avg_word_path",
    "phish_hints","domain_in_brand","brand_in_subdomain","brand_in_path","suspecious_tld","statistical_report",
    "nb_hyperlinks","ratio_intHyperlinks","ratio_extHyperlinks","ratio_nullHyperlinks","nb_extCSS",
    "ratio_intRedirection","ratio_extRedirection","ratio_intErrors","ratio_extErrors","login_form",
    "external_favicon","links_in_tags","submit_email","ratio_intMedia","ratio_extMedia","sfh","iframe",
    "popup_window","safe_anchor","onmouseover","right_clic","empty_title","domain_in_title","domain_with_copyright",
    "whois_registered_domain","domain_registration_length","domain_age","web_traffic","dns_record","google_index",
    "page_rank"
]

# Comparar listas para detectar diferencias

faltan_en_values = list(set(headers_list) - set(real_columns))
faltan_en_headers = list(set(values_list) - set(real_columns))

print("Faltan en values: ", faltan_en_values, "faltan en headers: ", faltan_en_headers)
