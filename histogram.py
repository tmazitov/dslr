import sys

import matplotlib.pyplot as plt

from data_utils import HOUSE_COLORS, HOUSE_ORDER, available_features, get_dataset_path, load_dataset


def homogeneity_score(df, feature):
    overall_std = df[feature].std()
    if not overall_std:
        return float('inf')
    house_means = df.groupby('Hogwarts House')[feature].mean()
    return house_means.std() / overall_std


def main():
    path = get_dataset_path()
    df = load_dataset(path)
    features = available_features(df)
    if not features:
        sys.exit("error: no numerical features found in dataset")

    scores = {feature: homogeneity_score(df, feature) for feature in features}
    best_feature = min(scores, key=scores.get)

    print("Homogeneity score by course (lower = more similar across houses):")
    for feature, score in sorted(scores.items(), key=lambda item: item[1]):
        print(f"  {feature:35s} {score:.4f}")
    print(f"\nMost homogeneous course across houses: {best_feature}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for house in HOUSE_ORDER:
        values = df.loc[df['Hogwarts House'] == house, best_feature].dropna()
        ax.hist(values, bins=25, alpha=0.6, label=house, color=HOUSE_COLORS[house])

    ax.set_title(f"{best_feature}: score distribution by house")
    ax.set_xlabel(best_feature)
    ax.set_ylabel("Number of students")
    ax.legend(title="House")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
