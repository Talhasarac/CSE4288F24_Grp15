import csv
from urllib.parse import urlparse, parse_qs

input_file = 'Csic_temiz.csv'
output_file = 'cleaned_output.csv'


patterns = {
    'has_index_jsp': 'index.jsp',
    'has_anadir_jsp': 'anadir.jsp',
    'has_pagar': 'pagar.jsp',
    'has_menum': 'menum.jsp',
    'has_titulo': 'titulo.jsp',
    'has_miembros': 'miembros',
    'has_estilos': 'estilos.css',
    'has_imagenes': 'imagenes',
    'has_caracter': 'carecteristicas.jsp',
    'has_creditos': 'creditos.jsp',
    "has_old": '.old',
    "has_nsf": '.nsf',
    "has_Bak": '.BAK',
    "has_auth": 'autenticar.jsp',
    "has_priv": '_private'
}

all_keys = []

with open("Csic_temiz.csv", 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter=';')
    for row in reader:
        url = row.get('URL', '')
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query) 
        all_keys.extend(params.keys())  


unique_keys = list(set(all_keys))
for key in unique_keys:
    patterns[f'has_{key}'] = key

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
   
    reader = csv.DictReader(infile, delimiter=';')
    fieldnames = reader.fieldnames + list(patterns.keys())
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    
    for row in reader:
        url = row.get('URL', '')
        
        for col, pat in patterns.items():
            row[col] = '1' if pat in url else '0'
        
        writer.writerow(row)

print("CSV cleaning complete. Results saved to:", output_file)
