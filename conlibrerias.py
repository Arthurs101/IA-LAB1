import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
file_path = 'C:/Users/marce/Desktop/DATA.csv'  
data = pd.read_csv(file_path)

# Seleccionar características y etiquetas
features = ['length_url', 'web_traffic']
X = data[features]
y = data['status']

# Convertir las etiquetas a valores numéricos
y = y.map({'legitimate': 0, 'phishing': 1})

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


class_distribution = y_train.value_counts()
if abs(class_distribution[0] - class_distribution[1]) > max(class_distribution) * 0.1:
    train_data = pd.DataFrame(X_train_scaled, columns=features)
    train_data['status'] = y_train.values
    legitimate = train_data[train_data.status == 0]
    phishing = train_data[train_data.status == 1]
    if class_distribution[0] > class_distribution[1]:
        phishing_upsampled = resample(phishing, replace=True, n_samples=class_distribution[0], random_state=42)
        train_data_balanced = pd.concat([legitimate, phishing_upsampled])
    else:
        legitimate_upsampled = resample(legitimate, replace=True, n_samples=class_distribution[1], random_state=42)
        train_data_balanced = pd.concat([legitimate_upsampled, phishing])
    X_train_balanced = train_data_balanced.drop('status', axis=1)
    y_train_balanced = train_data_balanced['status']
else:
    X_train_balanced = X_train_scaled
    y_train_balanced = y_train

# Implementar el algoritmo KNN
knn_balanced = KNeighborsClassifier(n_neighbors=3)
knn_balanced.fit(X_train_balanced, y_train_balanced)

# Realizar predicciones en el conjunto de prueba
y_pred_balanced = knn_balanced.predict(X_test_scaled)

# Calcular la precisión
accuracy_balanced = accuracy_score(y_test, y_pred_balanced)

# Graficar los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_scaled[:, 0], y=X_test_scaled[:, 1], hue=y_test, palette='coolwarm')
plt.title('KNN Classification Results (Balanced)')
plt.xlabel('Length of URL (Normalized)')
plt.ylabel('Web Traffic (Normalized)')
plt.legend(title='Status', labels=['Legitimate', 'Phishing'])
plt.show()

print(f'Accuracy: {accuracy_balanced}')
