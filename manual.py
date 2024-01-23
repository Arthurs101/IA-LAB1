import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
ruta_archivo = 'C:/Users/marce/Desktop/DATA.csv'
datos = pd.read_csv(ruta_archivo)

# Seleccionar dos variables y la variable objetivo
caracteristicas = ['length_url', 'web_traffic']
X = datos[caracteristicas].values.tolist()  # Convertir a listas para evitar usar NumPy
y = datos['status'].apply(lambda x: 0 if x == 'legitimate' else 1).values.tolist()

# Normalizar los datos
def normalizar(datos):
    minimo = min(datos)
    maximo = max(datos)
    return [(dato - minimo) / (maximo - minimo) for dato in datos]

X_norm = list(map(list, zip(*[normalizar(col) for col in zip(*X)])))

# Dividir los datos manualmente en entrenamiento y prueba (80/20)
tamaño_entrenamiento = int(0.8 * len(X_norm))
X_entrenamiento = X_norm[:tamaño_entrenamiento]
X_prueba = X_norm[tamaño_entrenamiento:]
y_entrenamiento = y[:tamaño_entrenamiento]
y_prueba = y[tamaño_entrenamiento:]

# Balanceo de clases 
from collections import Counter

def balancear_clases(X, y):
    contador = Counter(y)
    clase_minoritaria = min(contador, key=contador.get)
    clase_mayoritaria = max(contador, key=contador.get)
    diferencia = contador[clase_mayoritaria] - contador[clase_minoritaria]
    indices_minoritaria = [i for i, clase in enumerate(y) if clase == clase_minoritaria]
    nuevos_indices = indices_minoritaria * (diferencia // len(indices_minoritaria)) + indices_minoritaria[:diferencia % len(indices_minoritaria)]
    X += [X[i] for i in nuevos_indices]
    y += [y[i] for i in nuevos_indices]
    return X, y

X_entrenamiento, y_entrenamiento = balancear_clases(X_entrenamiento, y_entrenamiento)

# Función para calcular la distancia euclidiana 
def distancia_euclidiana(punto1, punto2):
    return sum((a - b) ** 2 for a, b in zip(punto1, punto2)) ** 0.5

# Función para encontrar los k vecinos más cercanos
def obtener_vecinos(X_entrenamiento, y_entrenamiento, instancia_prueba, k):
    distancias = []
    for i in range(len(X_entrenamiento)):
        dist = distancia_euclidiana(instancia_prueba, X_entrenamiento[i])
        distancias.append((dist, y_entrenamiento[i]))
    distancias.sort()
    vecinos = distancias[:k]
    return [vecino[1] for vecino in vecinos]

# Función para realizar predicciones con KNN
def predecir(X_entrenamiento, y_entrenamiento, X_prueba, k):
    predicciones = []
    for instancia in X_prueba:
        vecinos = obtener_vecinos(X_entrenamiento, y_entrenamiento, instancia, k)
        predicciones.append(max(set(vecinos), key=vecinos.count))
    return predicciones

# predicciones
k = 3
y_prediccion = predecir(X_entrenamiento, y_entrenamiento, X_prueba, k)

# precisión manualmente
precisión = sum(y_real == y_pred for y_real, y_pred in zip(y_prueba, y_prediccion)) / len(y_prueba)



for i, (caracteristica1, caracteristica2) in enumerate(X_prueba):
    plt.scatter(caracteristica1, caracteristica2, color='red' if y_prediccion[i] == 1 else 'green')
plt.xlabel(caracteristicas[0])
plt.ylabel(caracteristicas[1])
plt.title('Resultados de Clasificación KNN')
plt.show()

print(f'Precisión: {precisión}')
