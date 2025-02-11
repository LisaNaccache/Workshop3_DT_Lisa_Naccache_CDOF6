import requests
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# URLs des APIs des participants
api_urls = [
    "https://ec82-2a01-cb00-18d-a500-7050-6c92-4352-4208.ngrok-free.app/predict",
    "https://02ff-185-20-16-26.ngrok-free.app/predict"
]

# Fonction pour interroger une API et récupérer la prédiction
def get_prediction(api_url, features):
    try:
        response = requests.get(api_url, params=features)
        data = response.json()
        return data["predicted_class"]  # 🔥 Correction ici !
    except Exception as e:
        print(f"⚠️ Erreur avec {api_url}: {e}")
        return None

# Charger le dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Diviser en ensembles d'entraînement et de test
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

correct = 0
total = len(X_test)

for i, sample in enumerate(X_test):
    features = {
        "sepal_length": sample[0],
        "sepal_width": sample[1],
        "petal_length": sample[2],
        "petal_width": sample[3]
    }

    all_predictions = []
    for url in api_urls:
        result = get_prediction(url, features)
        if result is not None:  # 🔹 Vérifier que la prédiction est bien reçue
            all_predictions.append(result)

    if all_predictions:
        # 🟢 Vote majoritaire sur les classes (convertir en string pour éviter erreurs)
        final_prediction = max(set(all_predictions), key=all_predictions.count)

        # 🔹 Convertir les labels texte en indices numériques
        label_to_index = {"setosa": 0, "versicolor": 1, "virginica": 2}
        if final_prediction in label_to_index:
            final_prediction = label_to_index[final_prediction]

        if final_prediction == y_test[i]:
            correct += 1

# 📊 Calculer la précision du consensus
accuracy = correct / total
print(f"✅ Précision du méta-modèle de consensus : {accuracy:.2f}")
