import numpy as np
import pandas as pd
from data_utils import LOGREG_FEATURES, get_dataset_path, load_dataset, load_model

# ---------------------------
# main function
# ---------------------------
def main():
    # Dataset
    path = get_dataset_path()
    df = load_dataset(path)
    features = [f for f in LOGREG_FEATURES if f in df.columns]
    df[features] = df[features].fillna(df[features].mean())
    X = df[features].to_numpy()
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    W, B = load_model()
    
    # Predict
    predictions = predict_ova(X, W, B)

    reverse_mapping = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}

    houses = [reverse_mapping[p] for p in predictions]

    output = pd.DataFrame({
        "Index": np.arange(len(predictions)),
        "Hogwarts House": houses
    })

    output.to_csv("houses.csv", index=False)

# ---------------------------
# Sigmoid function
# ---------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------------------------
# Prediction
# ---------------------------
def predict_ova(X, W, B):

    scores = []

    for class_id in range(W.shape[0]):

        z = np.dot(X, W[class_id]) + B[class_id]
        probs = sigmoid(z)

        scores.append(probs)

    scores = np.array(scores).T

    # choose highest probability
    return np.argmax(scores, axis=1)

if __name__ == '__main__':
    main()