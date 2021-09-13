import unittest

from sklearn.datasets import load_digits, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier


class TestEvoCore(unittest.TestCase):

    def test_class(self):

        from meco import MECO

        X, y = load_digits(return_X_y=True)

        model = MECO(RidgeClassifier(random_state=42), 'both')
        model.fit(X, y)
        x_reduced = model.transform(X)
        y_reduced = y[model.best_set_['samples']]

        score1 = RidgeClassifier(random_state=42).fit(x_reduced, y_reduced).score(X[:, model.best_set_['features']], y)
        score2 = RandomForestClassifier(random_state=42).fit(x_reduced, y_reduced).score(X[:, model.best_set_['features']], y)

        print(score1)
        print(score2)
        print(model.best_set_)
        print(model.best_set_['accuracy'])

        return


suite = unittest.TestLoader().loadTestsFromTestCase(TestEvoCore)
unittest.TextTestRunner(verbosity=2).run(suite)
