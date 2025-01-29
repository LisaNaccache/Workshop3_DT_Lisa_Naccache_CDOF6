# train_model.py
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Charger le dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Séparer en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy du modèle SVM: {accuracy:.2f}")

# Sauvegarder le modèle et l'accuracy
with open("svm_iris_model.pkl", "wb") as f:
    pickle.dump({"model": model, "accuracy": accuracy}, f)

print("✅ Modèle entraîné et sauvegardé avec accuracy")
