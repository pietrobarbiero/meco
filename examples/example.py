import sys
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

from meco import MECO

random_state = 42
X1, y1 = make_classification(n_samples=2600, n_features=500,
                             n_informative=50, n_redundant=50, n_repeated=50,
                             class_sep=3, random_state=random_state, shuffle=False)
X2, y2 = make_classification(n_samples=2600, n_features=500,
                             n_informative=100, n_redundant=100, n_repeated=100,
                             class_sep=0.5, random_state=random_state, shuffle=False)
X3, y3 = make_classification(n_samples=2600, n_features=500,
                             n_informative=150, n_redundant=150, n_repeated=150,
                             class_sep=3, random_state=random_state, shuffle=False)


def main():
    X, y = X3, y3

    model = MECO(RidgeClassifier(random_state=42), 'both')
    model.fit(X, y)
    x_reduced = model.transform(X)
    y_reduced = y[model.best_set_['samples']]

    score1 = RidgeClassifier(random_state=42).fit(x_reduced, y_reduced).score(X[:, model.best_set_['features']], y)
    score2 = RandomForestClassifier(random_state=42).fit(x_reduced, y_reduced).score(X[:, model.best_set_['features']], y)

    print(score1)
    print(score2)
    print(model.best_set_['accuracy'])


if __name__ == "__main__":
    sys.exit(main())
