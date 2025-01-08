from urllib.parse import urlparse, parse_qs
import csv

all_keys = []

with open("Csic_temiz.csv", 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter=';')
    for row in reader:
        url = row.get('URL', '')
        parsed_url = urlparse(url)  # URL'i parse et
        params = parse_qs(parsed_url.query)  # Query kısmındaki parametreleri ayıkla
        all_keys.extend(params.keys())  # Anahtarları listeye ekle


# Tekrarlardan kurtulmak için set kullanarak benzersiz anahtarları listele
unique_keys = list(set(all_keys))

print(unique_keys)
