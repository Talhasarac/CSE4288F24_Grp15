import csv

# Input and output file paths
input_file = 'input.csv'
output_file = 'outputmethod.csv'

# Mapping for method
method_map = {
    'GET': '0',
    'POST': '1'
}

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile, delimiter=';')
    writer = csv.writer(outfile, delimiter=';')
    
    # Read the header row
    headers = next(reader)
    writer.writerow(headers)  # Write header as is
    
    # Find the index of the "Method" column
    method_index = headers.index("Method")
    
    # Process each subsequent row
    for row in reader:
        # Replace the value in the Method column using the mapping
        if row[method_index] in method_map:
            row[method_index] = method_map[row[method_index]]
        writer.writerow(row)
