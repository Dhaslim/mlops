# run_model.py — Test the model with custom input
import argparse, json
from pathlib import Path
import numpy as np, joblib


MODEL_PATH = Path("artifacts/model.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()


    features = json.loads(args.input)
    X = np.array(features).reshape(1, -1)
    model = joblib.load(MODEL_PATH)
    pred = model.predict(X)
    print(json.dumps({"prediction": pred.tolist()}))


if __name__ == "__main__":
    main()
