from sklearn.naive_bayes import GaussianNB, MultinomialNB

import utils
import numpy as np

def train_naive_bayes_numerical(train_x, train_y, test_x, test_y):
    model = GaussianNB()
    model.fit(train_x, train_y)
    accuracy = model.score(test_x, test_y)
    return (model, accuracy)


def train_naive_bayes_categorical(train_x, train_y, test_x, test_y):
    model = MultinomialNB()
    model.fit(train_x, train_y)
    accuracy = model.score(test_x, test_y)
    return (model, accuracy)


class CombinedNB:
    def __init__(self):
        self.numerical_model = GaussianNB()
        self.categorial_model = MultinomialNB()

    def predict(self, test_x, test_y=[]):
        prob_numerico = self.numerical_model.predict_proba(test_x)
        prob_categorico = self.categorial_model.predict_proba(test_x)
        
        prob_combined = prob_numerico * prob_categorico

        classes = self.numerical_model.classes_
        results = []
        for item in prob_combined:
            results.append(classes[np.argmax(item)])

        return np.array(results)

    def score(self, test_x, test_y):
        predictions = self.predict(test_x, test_y)
        return float((test_y == predictions).sum()) / float(len(predictions))

    def fit(self, train_x, train_y):
        self.numerical_model.fit(train_x, train_y)
        self.categorial_model.fit(train_x, train_y)


def train_naive_bayes_combined(train_x, train_y, test_x, test_y):
    model = CombinedNB()
    model.fit(train_x, train_y)
    accuracy = model.score(test_x, test_y)
    return model, accuracy


def old_main():
    data = utils.import_data()
    data = utils.scramble_data(data)

    train_data, test_data = utils.train_test_split(data)
    train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
    test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])

    (model, accuracy) = train_naive_bayes_numerical(utils.extract_numerical(train_x), train_y, utils.extract_numerical(test_x), test_y)
    print(f"Model accuracy [naive bayes numerical] = {accuracy}")
    (model, accuracy) = train_naive_bayes_categorical(utils.extract_categorical(train_x), train_y, utils.extract_categorical(test_x), test_y)
    print(f"Model accuracy [naive bayes categorical] = {accuracy}")


def main():
    data = utils.import_data()
    data = utils.scramble_data(data)

    train_data, test_data = utils.train_test_split(data)
    train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
    test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])

    (model, accuracy) = train_naive_bayes_combined(train_x, train_y, test_x, test_y)
    print(f"Model accuracy [naive bayes combined] = {accuracy}")


if __name__ == '__main__':
    main()


