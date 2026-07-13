import numpy as np

from data_utils import LOGREG_FEATURES, get_dataset_path, load_dataset


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
    mapping = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
    y = df['Hogwarts House'].map(mapping).to_numpy(dtype=int)

    # Train
    W, B = train_ova(
        X,
        y,
        num_classes=4,
        lr=0.1,
        epochs=5000
    )
    np.savez("logreg_model.npz", W=W, B=B)

# ---------------------------
# Sigmoid function
# ---------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------------------------
# Train one binary classifier
# ---------------------------
def train_binary_logistic(X, y, lr=0.1, epochs=1000):

    m, n = X.shape

    w = np.zeros(n)
    b = 0

    for _ in range(epochs):

        # Forward pass
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        # Gradients
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)
        # Update weights
        w -= lr * dw
        b -= lr * db

    return w, b

# ---------------------------
# One-vs-All training
# ---------------------------
def train_ova(X, y, num_classes, lr=0.1, epochs=1000):

    m, n = X.shape

    W = np.zeros((num_classes, n))
    B = np.zeros(num_classes)

    for class_id in range(num_classes):

        # Create binary labels
        y_binary = (y == class_id).astype(int)

        w, b = train_binary_logistic(
            X,
            y_binary,
            lr=lr,
            epochs=epochs
        )

        W[class_id] = w
        B[class_id] = b

    return W, B


if __name__ == '__main__':
    main()