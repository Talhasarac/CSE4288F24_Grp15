import csv

input_file = 'input.csv'
output_file = 'cleaned_output.csv'


patterns = {
    'has_index_jsp': 'index.jsp',
    'has_percent_login': '%login',
    'has_anadir_jsp': 'anadir.jsp',
    'has_entrar_login': 'entrar&login=',
    'has_pagar': 'pagar.jsp',
    'has_menum': 'menum.jsp',
    'has_titulo': 'titulo.jsp',
    'has_miembros': 'miembros',
    'has_estilos': 'estilos.css',
    'has_imagenes': 'imagenes',
    'has_caracter': 'carecteristicas.jsp',
    'has_side': '?id=',
    'has_creditos': 'creditos.jsp',
    'has_creditos': 'creditos.jsp',
    'has_pwd': 'pwd=',
    "has_login": 'login=',
    "has_pass": 'password=',
    "has_old": '.old',
    "has_nsf": '.nsf',
    "has_B1": 'B1=',
    "has_Bak": '.BAK',
    "has_auth": 'autenticar.jsp',
    "has_modo": 'modo=insertar',
    "has_priv": '_private'
    
}

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
