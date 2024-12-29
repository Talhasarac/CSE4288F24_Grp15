import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle


file_path = 'cleaned_output.csv'
data = pd.read_csv(file_path, sep=';', engine='python')
data = data.drop(columns=['host', 'lenght', 'content', 'content-type', 'URL'])


print(data.head())


label_encoders = {}
for column in ['Method', 'has_index_jsp', 'has_percent_login', 'has_anadir_jsp', 'has_entrar_login', 'has_pagar',
               'has_menum', 'has_titulo', 'has_miembros', 'has_estilos', 'has_imagenes',
               'has_caracter', 'has_side', 'has_creditos', 'has_pwd', "has_login",
               "has_pass", "has_old", "has_nsf", "has_B1", "has_Bak", "has_auth", "has_modo", "has_priv"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le


X = data.drop(columns=['classification'])
y = data['classification']


label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
print("NORMAL = 1, ATTACK = 0")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")  


print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))


import pickle


model_filename = 'naive_bayes_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)
    print(f"Model saved as '{model_filename}'")


label_encoders_filename = 'label_encoders.pkl'
with open(label_encoders_filename, 'wb') as encoders_file:
    pickle.dump(label_encoders, encoders_file)
    print(f"Feature label encoders saved as '{label_encoders_filename}'")


target_encoder_filename = 'label_encoder_y.pkl'
with open(target_encoder_filename, 'wb') as target_encoder_file:
    pickle.dump(label_encoder_y, target_encoder_file)
    print(f"Target label encoder saved as '{target_encoder_filename}'")



import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cm = confusion_matrix(y_test, y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder_y.classes_)


plt.figure(figsize=(8, 6))
disp.plot(cmap='viridis', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.jpg')
plt.show()
