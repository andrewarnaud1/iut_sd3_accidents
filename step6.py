import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Charger les données
data = pd.read_csv("step5/one_hot_encoding.csv")

# Séparer les caractéristiques et la cible
X = data.drop("grav", axis=1)
y = data["grav"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Fonction pour évaluer un modèle
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    return accuracy, precision, recall, f1


# Tester différents modèles
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
}

best_score = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print(
        f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
    )

    # Sélectionner le meilleur modèle
    if accuracy > best_score:
        best_score = accuracy
        best_model = model

# Sauvegarder le meilleur modèle
with open("step6/best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Exporter les échantillons train et test
X_train.to_csv("step6/train.csv", index=False)
X_test.to_csv("step6/test.csv", index=False)
