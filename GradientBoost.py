import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
file_path = 'cleaned_output.csv'  # Replace with the actual file path
data = pd.read_csv(file_path, sep=';', engine='python')

# Drop unnecessary columns
data = data.drop(columns=['host', 'lenght', 'content', 'content-type', 'URL'])

# Display the first few rows to understand the structure
print(data.head())

# Encode categorical columns
label_encoders = {}
categorical_columns = [
    'Method', 'has_index_jsp', 'has_percent_login', 'has_anadir_jsp', 'has_entrar_login',
    'has_pagar', 'has_menum', 'has_titulo', 'has_miembros', 'has_estilos',
    'has_imagenes', 'has_caracter', 'has_side', 'has_creditos', 'has_pwd',
    'has_login', 'has_pass', 'has_old', 'has_nsf', 'has_B1', 'has_Bak', 
    'has_auth', 'has_modo', 'has_priv'
]

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Define features and target
X = data.drop(columns=['classification'])
y = data['classification']

# Encode the target variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
print("NORMAL = 1, ATTACK = 0")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance_df)

# Save the model and label encoders
with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('label_encoder_y.pkl', 'wb') as f:
    pickle.dump(label_encoder_y, f)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create the Confusion Matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder_y.classes_)

# Plot and save the confusion matrix as an image
plt.figure(figsize=(8, 6))
disp.plot(cmap='viridis', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.jpg')
plt.show()
