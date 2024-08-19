import pandas as pd
import numpy as np
import pprint

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


def main():
    # Import and scramble data
    data = utils.import_data()
    data = utils.scramble_data(data)
    # data.to_csv('all_data.csv', index=False)
    # data.to_csv('all_data.csv', index=False)

    # Split into train and test data
    train_data, val_data, test_data = utils.stratified_split(data)

    # Split into x and y columns
    train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
    test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])
    val_x, val_y = utils.xy_split(val_data, columns=['Weather Type'])

    models = build_models(train_x, train_y, test_x, test_y)

    print('ACURACIAS OBTIDAS COM OTIMIZACAO DE HIPERPARAMETROS COM O CONJUNTO DE VALIDACAO:\n')
    for model in models:
        model_label = model['label']
        model_obj = model['model']
        accuracy = model_obj.score(val_x, val_y)
        print(f'Modelo "{model_label}":\n{accuracy}')

    #pprint.pprint(models)

    print('\nACURACIAS OBTIDAS APOS OTIMIZACAO DE HIPERPARAMETROS COM O CONJUNTO DE TESTE:\n')
    print('Acuracia do modelo Arvore de Decisao:\n', models[0]['accuracy'])
    print('Acuracia do modelo kNN com distancia Euclidiana:\n', models[1]['accuracy'],' k = ',models[1]['model'].n_neighbors)
    print('Acuracia do modelo kNN com distancia Chebyshev:\n', models[2]['accuracy'],' k = ',models[2]['model'].n_neighbors)
    print('Acuracia do modelo kNN com distancia Manhattan:\n', models[3]['accuracy'],' k = ',models[3]['model'].n_neighbors)
    print('Acuracia do modelo Naive Bayes:\n', models[4]['accuracy'],'\n')


if __name__ == '__main__':
    main()
