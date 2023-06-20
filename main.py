import numpy as np
from sklearn.linear_model import LogisticRegression

class AdaptiveThresholdClassifier:
    def __init__(self, threshold_initial=0.5, threshold_increment=0.1):
        self.threshold = threshold_initial
        self.threshold_increment = threshold_increment
        self.model = LogisticRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions

    def update_threshold(self, X_train, y_train):
        predictions = self.model.predict(X_train)
        incorrect_predictions = predictions != y_train
        self.threshold += self.threshold_increment * incorrect_predictions.mean()

# Datos de entrenamiento flores Iris para el entrenamiento
X_train = np.array([[5.1, 3.5, 1.4, 0.2],
                    [4.9, 3.0, 1.4, 0.2],
                    [6.2, 2.2, 4.5, 1.5],
                    [5.7, 2.8, 4.1, 1.3],
                    [6.3, 2.7, 4.9, 1.8],
                    [7.6, 3.0, 6.6, 2.1],
                    [6.8, 2.8, 4.8, 1.4],
                    [7.7, 2.6, 6.9, 2.3]])

y_train = np.array([0, 0, 1, 1, 1, 2, 1, 2])

# Datos de prueba
X_test = np.array([[5.5, 3.4, 1.2, 0.3],
                   [6.1, 2.9, 4.7, 1.4],
                   [7.3, 2.9, 6.3, 1.8]])

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
