import sys

import matplotlib.pyplot as plt

from data_utils import HOUSE_COLORS, HOUSE_ORDER, available_features, get_dataset_path, load_dataset


def most_similar_pair(df, features):
    corr = df[features].corr()
    best_value, best_a, best_b = 0.0, None, None
    for i, feat_a in enumerate(features):
        for feat_b in features[i + 1:]:
            value = abs(corr.loc[feat_a, feat_b])
            if value > best_value:
                best_value, best_a, best_b = value, feat_a, feat_b
    return best_value, best_a, best_b


def main():
    path = get_dataset_path()
    df = load_dataset(path)
    features = available_features(df)
    if len(features) < 2:
        sys.exit("error: need at least two numerical features to compare")

    score, feat_a, feat_b = most_similar_pair(df, features)
    print(f"Most similar features: {feat_a} and {feat_b} (|r| = {score:.4f})")

    fig, ax = plt.subplots(figsize=(7, 6))
    for house in HOUSE_ORDER:
        subset = df[df['Hogwarts House'] == house]
        ax.scatter(subset[feat_a], subset[feat_b], s=14, alpha=0.6,
                   label=house, color=HOUSE_COLORS[house])

    ax.set_title(f"{feat_a} vs {feat_b}")
    ax.set_xlabel(feat_a)
    ax.set_ylabel(feat_b)
    ax.legend(title="House")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
