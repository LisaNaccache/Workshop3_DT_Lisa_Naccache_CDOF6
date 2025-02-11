import requests
import numpy as np
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🛠️ Charger les poids des modèles depuis `database.json`
with open("database.json", "r") as f:
    database = json.load(f)

# 🏆 Construire un dictionnaire des poids
model_weights = {model["id"]: model["weight"] for model in database["models"]}

# 🔗 URLs des APIs des participants (sans paramètres)
api_urls = {
    "lisa": "https://ec82-2a01-cb00-18d-a500-7050-6c92-4352-4208.ngrok-free.app/predict",
    "leina": "https://02ff-185-20-16-26.ngrok-free.app/predict"
}

# Fonction pour interroger une API et récupérer la prédiction
def get_prediction(api_url, features):
    try:
        response = requests.get(api_url, params=features)
        data = response.json()
        return data["predicted_class"]
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
model_correct_counts = {model_id: 0 for model_id in api_urls}  # Comptabiliser les bonnes réponses par modèle

for i, sample in enumerate(X_test):
    features = {
        "sepal_length": sample[0],
        "sepal_width": sample[1],
        "petal_length": sample[2],
        "petal_width": sample[3]
    }

    weighted_predictions = {}  # Stocker les prédictions pondérées
    sum_weights = 0

    for model_id, url in api_urls.items():
        result = get_prediction(url, features)
        if result is not None:
            # ⚡ Convertir le nom de classe en entier (ex: "setosa" -> 0)
            class_index = list(iris.target_names).index(result)

            # 🔢 Appliquer la pondération
            weight = model_weights.get(model_id, 1.0)  # Valeur par défaut = 1.0 si le modèle n'existe pas dans database.json
            weighted_predictions[class_index] = weighted_predictions.get(class_index, 0) + weight
            sum_weights += weight

            # 📊 Suivi des bonnes réponses
            if class_index == y_test[i]:
                model_correct_counts[model_id] += 1

    if weighted_predictions:
        # 🏆 Déterminer la classe avec le score pondéré le plus élevé
        final_prediction = max(weighted_predictions, key=weighted_predictions.get)
        if final_prediction == y_test[i]:
            correct += 1

# 📊 Calculer la nouvelle précision
accuracy = correct / total
print(f"✅ Précision du méta-modèle de consensus pondéré : {accuracy:.2f}")

# 🔄 Mise à jour des poids des modèles en fonction des bonnes prédictions
for model_id in model_correct_counts:
    if total > 0:
        model_weights[model_id] = model_correct_counts[model_id] / total

# 📂 Sauvegarder les nouveaux poids dans `database.json`
for model in database["models"]:
    if model["id"] in model_weights:
        model["weight"] = model_weights[model["id"]]

with open("database.json", "w") as f:
    json.dump(database, f, indent=4)

print("📌 Mise à jour des poids enregistrée dans `database.json` ✅")