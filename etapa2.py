import pandas as pd
import numpy as np
import pprint

import utils
from decision_tree import train_decision_tree
from knn import train_knn_euc, train_knn_che, train_knn_man
from naive_bayes import train_naive_bayes_combined
import os


def build_models(train_x, train_y, test_x, test_y):
    training_parameters = [
        { 'label': 'decision_tree',         'training_function': train_decision_tree,           'input_function': utils.identity            },
        { 'label': 'knn_euc',               'training_function': train_knn_euc,                 'input_function': utils.identity            },
        { 'label': 'knn_che',               'training_function': train_knn_che,                 'input_function': utils.identity            },
        { 'label': 'knn_man',               'training_function': train_knn_man,                 'input_function': utils.identity            },
        { 'label': 'naive_bayes',           'training_function': train_naive_bayes_combined,    'input_function': utils.identity            },
    ]

    models = []
    
    print('\nIniciando treinamento dos modelos...\n')
    for param in training_parameters:
        label = param['label']
        training_function = param['training_function']
        input_function = param['input_function']

        print(f'Treinando modelo \'{label}\'')
        (model, accuracy) = training_function(input_function(train_x), train_y, input_function(test_x), test_y)
        models.append({ 'label': label, 'model': model, 'accuracy': accuracy, 'input_function': input_function })
    print('\nTreinamento dos modelos concluido com sucesso.\n')

    return models


def run_test(data, root_dir, *, 
             remove_outliers=True,
             train_data_strategy='holdout'
             ):
    # Data pre-processing
    if remove_outliers:
        data = utils.remove_outliers(data)
        data.to_csv('outlier_free_data.csv', index=False)
    data.to_csv('raw_data.csv', index=False)
    data = utils.one_hot_encoding(data)
    data = utils.normalizar_dados(data)
    data = utils.scramble_data(data)
    data.to_csv('clean_data.csv', index=False)

    # Data splitting
    if train_data_strategy == 'holdout':
        train_data, val_data, test_data = utils.stratified_split(data)
        train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
        test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])
        val_x, val_y = utils.xy_split(val_data, columns=['Weather Type'])
    elif train_data_strategy == 'bootstrap':
        train_data, val_data, test_data = utils.bootstrap(data)
        train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
        test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])
        val_x, val_y = utils.xy_split(val_data, columns=['Weather Type'])
    else:
        raise ValueError(f'Cannot run test with parameter train_data_strategy set to "{train_data_strategy}"')

    # Model building
    models = build_models(train_x, train_y, test_x, test_y)

    # Model evaluation
    for model in models:
        model_label = model['label']
        model_obj = model['model']
        accuracy = model_obj.score(val_x, val_y)
        model['validation_accuracy'] = accuracy

    for model in models:
        utils.makeRocCurve(model_obj, f'train_{model_label}', test_x, test_y, train_y, root_dir)
        utils.makePrCurve(model_obj, f'train_{model_label}', test_x, test_y, train_y, root_dir)
    
    return models


def main():
    # Import data
    filename = 'weather_classification_data.csv'
    data = utils.ler_dados_csv(filename)

    result_set = []

    strategies = [
            'holdout',
            'bootstrap',
            ]

    test_index = 0
    for remove_outliers in range(2):
        for test_strategy in strategies:
            test_index += 1
            test_directory = f'test_{test_index}'
            os.makedirs(test_directory, exist_ok=True)
            with open(f'{test_directory}/description.txt', 'w') as file:
                file.write(f'[parameters]\n')
                file.write(f'remove_outliers={bool(remove_outliers)}\n')
                file.write(f'test_strategy={test_strategy}\n')

            models = run_test(data, test_directory,
                              remove_outliers=bool(remove_outliers),
                              train_data_strategy=test_strategy,
                              )
            result_set.append({
                'test_index': test_index,
                'remove_outliers': bool(remove_outliers),
                'test_strategy': test_strategy,
                'models': models,
                })

    for test_data in result_set:
        pprint.pprint(test_data)


if __name__ == '__main__':
    main()

