import requests
import numpy as np
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 🛠️ Charger les données depuis `database.json`
with open("database.json", "r") as f:
    database = json.load(f)

# 🏆 Construire un dictionnaire des poids et soldes
model_weights = {model["id"]: model["weight"] for model in database["models"]}
model_balances = {model["id"]: model["balance"] for model in database["models"]}

# 🔗 URLs des APIs des participants
api_urls = {
    "lisa": "https://44e6-89-30-29-68.ngrok-free.app/predict",
    "leina": "https://513c-89-30-29-68.ngrok-free.app/predict"
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
model_correct_counts = {model_id: 0 for model_id in api_urls}  # Suivi des bonnes réponses par modèle
model_error_counts = {model_id: 0 for model_id in api_urls}  # Suivi des erreurs

for i, sample in enumerate(X_test):
    features = {
        "sepal_length": sample[0],
        "sepal_width": sample[1],
        "petal_length": sample[2],
        "petal_width": sample[3]
    }

    weighted_predictions = {}
    sum_weights = 0
    all_predictions = []

    for model_id, url in api_urls.items():
        result = get_prediction(url, features)
        if result is not None:
            class_index = list(iris.target_names).index(result)
            all_predictions.append(class_index)

            # 🔢 Appliquer la pondération
            weight = model_weights.get(model_id, 1.0)
            weighted_predictions[class_index] = weighted_predictions.get(class_index, 0) + weight
            sum_weights += weight

            # 📊 Mise à jour des scores
            if class_index == y_test[i]:
                model_correct_counts[model_id] += 1
            else:
                model_error_counts[model_id] += 1  # Suivi des erreurs

    if weighted_predictions:
        final_prediction = max(weighted_predictions, key=weighted_predictions.get)
        if final_prediction == y_test[i]:
            correct += 1

# 📊 Calculer la précision
accuracy = correct / total
print(f"✅ Précision du méta-modèle de consensus pondéré : {accuracy:.2f}")

# 🔄 Mise à jour des poids des modèles en fonction des bonnes prédictions
for model_id in model_correct_counts:
    if total > 0:
        model_weights[model_id] = model_correct_counts[model_id] / total

# 🏆 Appliquer le Slashing (Pénalisation des erreurs)
PENALTY = 10  # Montant de l'amende par erreur
REWARD = 5  # Bonus pour ceux qui ont bien prédit

for model_id in model_error_counts:
    errors = model_error_counts[model_id]

    # ⚠️ Si un modèle fait trop d'erreurs, il est pénalisé financièrement
    if errors > 0:
        model_balances[model_id] -= errors * PENALTY
        print(f"❌ {model_id} a perdu {errors * PENALTY}€ (Slashing)")

    # 🎉 Bonus pour ceux qui prédisent bien
    if model_correct_counts[model_id] > 0:
        model_balances[model_id] += model_correct_counts[model_id] * REWARD
        print(f"💰 {model_id} gagne {model_correct_counts[model_id] * REWARD}€")

# 📂 Sauvegarder les nouveaux poids et soldes dans `database.json`
for model in database["models"]:
    model["weight"] = model_weights[model["id"]]
    model["balance"] = model_balances[model["id"]]

with open("database.json", "w") as f:
    json.dump(database, f, indent=4)

print("📌 Mise à jour des poids et des soldes enregistrée dans `database.json` ✅")
print(f"📊 Soldes mis à jour : {model_balances}")
