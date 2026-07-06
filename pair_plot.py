import matplotlib.pyplot as plt

from data_utils import HOUSE_COLORS, HOUSE_ORDER, LOGREG_FEATURES, get_dataset_path, load_dataset


def main():
    path = get_dataset_path()
    df = load_dataset(path)
    features = [f for f in LOGREG_FEATURES if f in df.columns]

    print("Features for logistic regression:")
    for feature in features:
        print(f"  - {feature}")

    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))

    for row, feat_y in enumerate(features):
        for col, feat_x in enumerate(features):
            ax = axes[row][col]
            if row == col:
                for house in HOUSE_ORDER:
                    values = df.loc[df['Hogwarts House'] == house, feat_x].dropna()
                    ax.hist(values, bins=15, alpha=0.6, color=HOUSE_COLORS[house])
            else:
                for house in HOUSE_ORDER:
                    subset = df[df['Hogwarts House'] == house]
                    ax.scatter(subset[feat_x], subset[feat_y], s=5, alpha=0.5,
                               color=HOUSE_COLORS[house])

            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(feat_y, fontsize=8, rotation=0, ha='right', va='center')
            if row == n - 1:
                ax.set_xlabel(feat_x, fontsize=8, rotation=90)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=HOUSE_COLORS[house],
                   markersize=8, label=house)
        for house in HOUSE_ORDER
    ]
    fig.legend(handles=handles, loc='upper right', title="House")
    fig.suptitle("Pair plot of selected Hogwarts course scores")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


if __name__ == '__main__':
    main()
