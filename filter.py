import pandas as pd

# Load the CSV file with semicolon as delimiter
df = pd.read_csv('Csic_temiz.csv', delimiter=';')

# Print the column names to check the correct column name
print(df.columns)

# Filter rows where classification is 1
filtered_df = df[df['classification'] == 1]

# Write the filtered rows to class1.txt
filtered_df.to_csv('class1.txt', index=False, header=False)