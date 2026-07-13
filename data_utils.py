import sys
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    'Arithmancy',
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Potions',
    'Care of Magical Creatures',
    'Charms',
    'Flying',
]

LOGREG_FEATURES = [
    'Herbology',
    'Defense Against the Dark Arts',
    'Muggle Studies',
    'Charms',
    'Flying',
]

HOUSE_ORDER = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

HOUSE_COLORS = {
    'Gryffindor': "#d6302a",
    'Hufflepuff': "#e9d627",
    'Ravenclaw': "#862ad6",
    'Slytherin': '#008300',
}


def get_dataset_path(default='dataset_train.csv'):
    argv = sys.argv
    if len(argv) > 2:
        sys.exit("error: invalid number of arguments")
    return argv[1] if len(argv) == 2 else default


def load_dataset(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"error: file '{path}' not found")
    except pd.errors.EmptyDataError:
        sys.exit("error: dataset is empty")
    except OSError as e:
        sys.exit(f"error: {e}")

    if 'Hogwarts House' not in df.columns:
        sys.exit("error: dataset is missing the 'Hogwarts House' column")

    return df

def load_model():
    try:
        data = np.load("logreg_model.npz")
        W = data["W"]
        B = data["B"]
    except FileNotFoundError:
        print("Error: Model file 'logreg_model.npz' not found.")
        sys.exit(1)
    except KeyError:
        print("Error: Model file is missing 'W' or 'B'.")
        sys.exit(1)
    except (ValueError, EOFError, OSError):
        print("Error: Model file is empty, corrupted, or invalid.")
        sys.exit(1)
    return W, B


def available_features(df):
    return [f for f in NUMERIC_FEATURES if f in df.columns]
