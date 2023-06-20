from sklearn.linear_model import LogisticRegression

class AdaptiveThresholdClassifier:
    def __init__(self, threshold_initial=0.5, threshold_increment=0.1):
        self.threshold = threshold_initial
        self.threshold_increment = threshold_increment
        self.model = LogisticRegression()  # Modelo de regresión logística

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        probabilities = self.model.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions

    def update_threshold(self, X_train, y_train):
        predictions = self.model.predict(X_train)
        incorrect_predictions = predictions != y_train
        self.threshold += self.threshold_increment * incorrect_predictions.mean()

# Ejemplo de uso
# Datos de entrenamiento
X_train = [[0.1], [0.3], [0.5], [0.7], [0.9]]
y_train = [0, 0, 1, 1, 1]

# Datos de prueba
X_test = [[0.2], [0.4], [0.6], [0.8]]

# Crear y entrenar el clasificador con umbrales adaptativos
classifier = AdaptiveThresholdClassifier()
classifier.fit(X_train, y_train)

# Realizar predicciones
predictions = classifier.predict(X_test)
print("Predicciones:", predictions)

# Actualizar los umbrales adaptativos con los datos de entrenamiento
classifier.update_threshold(X_train, y_train)

# Realizar nuevas predicciones con los umbrales actualizados
predictions = classifier.predict(X_test)
print("Nuevas predicciones:", predictions)
