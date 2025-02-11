# api.py

from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn import datasets

app = Flask(__name__)

# Charger le dataset Iris pour retrouver les noms des classes
iris = datasets.load_iris()

# Charger le modèle et l'accuracy
with open("svm_iris_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    accuracy = data["accuracy"]  # Accuracy sauvegardée

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Récupérer les paramètres depuis l'URL
        sepal_length = float(request.args.get('sepal_length', 5.1))
        sepal_width = float(request.args.get('sepal_width', 3.5))
        petal_length = float(request.args.get('petal_length', 1.4))
        petal_width = float(request.args.get('petal_width', 0.2))

        # Créer une entrée pour le modèle
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Faire la prédiction
        prediction = model.predict(input_data)[0]  # Classe prédite
        class_name = iris.target_names[prediction]  # Nom de la classe (Setosa, Versicolor, Virginica)

        response = {
            "model_name": "SVM",
            "predicted_class": class_name,
            "accuracy": round(accuracy, 2)  # Formatté à 2 décimales
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
