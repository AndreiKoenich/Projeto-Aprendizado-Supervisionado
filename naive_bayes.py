from sklearn.naive_bayes import GaussianNB, MultinomialNB

import utils

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


def main():
    data = utils.import_data()
    data = utils.scramble_data(data)

    train_data, test_data = utils.train_test_split(data)
    train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
    test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])

    (model, accuracy) = train_naive_bayes_numerical(utils.extract_numerical(train_x), train_y, utils.extract_numerical(test_x), test_y)
    print(f"Model accuracy [naive bayes numerical] = {accuracy}")
    (model, accuracy) = train_naive_bayes_categorical(utils.extract_categorical(train_x), train_y, utils.extract_categorical(test_x), test_y)
    print(f"Model accuracy [naive bayes categorical] = {accuracy}")


if __name__ == '__main__':
    main()


