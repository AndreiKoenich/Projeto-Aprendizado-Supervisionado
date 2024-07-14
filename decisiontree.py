from sklearn import tree
import matplotlib.pyplot as plt
from utils import cria_instancia_teste, ler_dados_csv, normalizar_dados, one_hot_encoding, cross_validation
import pandas as pd


def aplicar_arvore_decisao(dados_normalizados):
    # Separar atributo alvo
    atributo_alvo = dados_normalizados['Weather Type']
    
    # Criar modelo
    clf = tree.DecisionTreeClassifier()
    
    # Treinar o modelo com os dados normalizados
    clf.fit(dados_normalizados.drop(columns=['Weather Type']), atributo_alvo)
    
    # Plotar a árvore de decisão
    # plt.figure(figsize=(20,10))
    # tree.plot_tree(clf, filled=True, feature_names=dados_normalizados.columns[:-1], class_names=clf.classes_)
    # plt.show()
    
    return clf


def evaluate_model(model, test_data):
    # Split test data
    target = test_data['Weather Type']
    data = test_data.drop(columns=['Weather Type'])

    # Predict results
    prediction = model.predict(data)
    
    # Calculate accuracy
    accuracy = float((prediction == target).sum()) / float(len(prediction))

    return accuracy


def main():
    nome_arquivo = 'weather_classification_data.csv'
    dados = ler_dados_csv(nome_arquivo)
    dados_encoded = one_hot_encoding(dados)
    dados_normalizados = normalizar_dados(dados_encoded)

    k = 5 # For k-fold cross-validation
    for left, right in cross_validation(len(dados_normalizados), k):
        # Embaralhar os dados e dividir em fracoes de teste e treinamento
        dados_normalizados = dados_normalizados.sample(frac=1).reset_index(drop=True)
        training_data = pd.concat((dados_normalizados[0:left], dados_normalizados[right:]))
        test_data = dados_normalizados[left:right]

        # Treinar o modelo
        trained_model = aplicar_arvore_decisao(training_data)

        # Avaliar o modelo
        results = evaluate_model(trained_model, test_data)
        # print(results)


main()

