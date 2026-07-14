# train.py
from sklearn.datasets import load_iris          # Built-in dataset
from sklearn.linear_model import LogisticRegression  # The algorithm
from sklearn.model_selection import train_test_split # Split data 80/20
import joblib, os, json


def main():
    iris = load_iris()
    X, y = iris.data, iris.target   # X = features, y = labels (0,1,2)


    # Split 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Choose algorithm and train
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)


    # Save the trained model to disk
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")


    # Save accuracy metrics
    acc = model.score(X_test, y_test)
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": float(acc)}, f)


    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

