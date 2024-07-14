from sklearn.neighbors import KNeighborsClassifier
from utils import cria_instancia_teste, ler_dados_csv, normalizar_dados, one_hot_encoding
import numpy as np


def aplicar_knn(k, dados_normalizados, instancia_teste, metrica):
    # Separar atributo alvo
    atributo_alvo = dados_normalizados['Weather Type']
    
    # Criar modelo kNN
    knn = KNeighborsClassifier(n_neighbors=k, metric=metrica)
    
    # Treinar o modelo com os dados normalizados
    knn.fit(dados_normalizados.drop(columns=['Weather Type']), atributo_alvo)
    
    # Realizar previsão para a instância de teste
    resultado = knn.predict(instancia_teste.drop(columns=['Weather Type']))
    
    # Retorna o valor do atributo alvo para a instância de teste
    return resultado 

def main():
    nome_arquivo = 'weather_classification_data.csv'
    dados = ler_dados_csv(nome_arquivo)
    dados_encoded = one_hot_encoding(dados)
    dados_normalizados = normalizar_dados(dados_encoded)

    #print(dados)
    #print(dados_encoded)
    #print(dados_normalizados)

    # Shuffle data
    dados_normalizados = dados_normalizados.sample(frac=1).reset_index(drop=True)

    # Split into test data (15%) and model data (85%)
    test_size = int(0.15 * len(dados_normalizados))
    test_data = dados_normalizados[0:test_size]
    model_data = dados_normalizados[test_size:]

    # Compute target column
    target = test_data['Weather Type']

    print('\n### RESULTADOS COM DISTANCIA EUCLIDIANA ###')
    results = []
    for k in range(1, 51, 2):
        prediction = aplicar_knn(k, model_data, test_data, 'euclidean')
        accuracy = float((prediction == target).sum()) / float(len(prediction))
        # print(accuracy,"\tk = ", k, sep="")
        results.append((accuracy, k))
    best_result = max(results, key=lambda x: x[0])
    print("Melhor resultado com k=", best_result[1], ", com acuracia de: ", 100*best_result[0], "%", sep="")

    print('\n### RESULTADOS COM DISTANCIA DE CHEBYSHEV ###')
    results = []
    for k in range(1, 51, 2):
        prediction = aplicar_knn(k, model_data, test_data, 'chebyshev')
        accuracy = float((prediction == target).sum()) / float(len(prediction))
        # print(accuracy,"\tk = ", k, sep="")
        results.append((accuracy, k))
    best_result = max(results, key=lambda x: x[0])
    print("Melhor resultado com k=", best_result[1], ", com acuracia de: ", 100*best_result[0], "%", sep="")

    print('\n### RESULTADOS COM DISTANCIA MANHATTAN ###')
    for k in range(1, 51, 2):
        prediction = aplicar_knn(k, model_data, test_data, 'manhattan')
        accuracy = float((prediction == target).sum()) / float(len(prediction))
        # print(accuracy,"\tk = ", k, sep="")
        results.append((accuracy, k))
    best_result = max(results, key=lambda x: x[0])
    print("Melhor resultado com k=", best_result[1], ", com acuracia de: ", 100*best_result[0], "%", sep="")


main()

