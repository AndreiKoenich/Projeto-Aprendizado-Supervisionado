import pprint

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import utils
from decision_tree import train_decision_tree
from knn import train_knn_euc, train_knn_che, train_knn_man
from naive_bayes import train_naive_bayes_combined


# ORCHESTRATION
def build_models(train_x, train_y, test_x, test_y):
    training_parameters = [
        { 'label': 'decision_tree',         'training_function': train_decision_tree,           'input_function': utils.identity            },
        { 'label': 'knn_euc',               'training_function': train_knn_euc,                 'input_function': utils.identity            },
        { 'label': 'knn_che',               'training_function': train_knn_che,                 'input_function': utils.identity            },
        { 'label': 'knn_man',               'training_function': train_knn_man,                 'input_function': utils.identity            },
        { 'label': 'naive_bayes',           'training_function': train_naive_bayes_combined,    'input_function': utils.identity            },
    ]

    models = []

    for param in training_parameters:
        label = param['label']
        training_function = param['training_function']
        input_function = param['input_function']

        print(f'Training model \'{label}\'')
        (model, accuracy) = training_function(input_function(train_x), train_y, input_function(test_x), test_y)
        models.append({ 'label': label, 'model': model, 'accuracy': accuracy, 'input_function': input_function })

    return models

def input_space_to_prediction_space(models, X):
    predictions = []

    for _model in models:
        label = _model['label']
        model = _model['model']
        input_function = _model['input_function']

        print(f'Prediction with model \'{label}\'')
        prediction = model.predict(input_function(X))
        predictions.append(prediction)

    predictions = np.array(predictions)
    predictions = predictions.transpose()
    predictions = pd.DataFrame(predictions)
    predictions.columns = [model['label'] for model in models]
    return predictions


def train_stacking_model(train_x, train_y, test_x, test_y):
    model = MultinomialNB()
    model.fit(train_x, train_y)
    accuracy = model.score(test_x, test_y)
    return (model, accuracy)


def main():
    # Import and scramble data
    data = utils.import_data()
    data = utils.scramble_data(data)
    # Split into train and test data
    train_data, test_data = utils.train_test_split(data)
    # Split into x and y columns
    train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
    test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])

    models = build_models(train_x, train_y, test_x, test_y)

    pprint.pprint(models)

    train_predictions = input_space_to_prediction_space(models, train_x)
    train_predictions = utils._one_hot_encoding(train_predictions, columns=[model['label'] for model in models])
    test_predictions = input_space_to_prediction_space(models, test_x)
    test_predictions = utils._one_hot_encoding(test_predictions, columns=[model['label'] for model in models])

    (model, accuracy) = train_stacking_model(train_predictions, train_y, test_predictions, test_y)
    print(accuracy)



if __name__ == '__main__':
    main()

