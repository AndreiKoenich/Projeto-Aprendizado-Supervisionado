from sklearn.neighbors import KNeighborsClassifier
from utils import cria_instancia_teste, ler_dados_csv, normalizar_dados, one_hot_encoding


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
    return resultado[0]  

def main():
    nome_arquivo = 'weather_classification_data.csv'
    dados = ler_dados_csv(nome_arquivo)
    dados_encoded = one_hot_encoding(dados)
    dados_normalizados = normalizar_dados(dados_encoded)

    #print(dados)
    #print(dados_encoded)
    #print(dados_normalizados)

    instancia_teste = cria_instancia_teste()

    print('\n### RESULTADOS COM DISTANCIA EUCLIDIANA ###\n')
    for k in range(1, 101, 2):
        resultado = aplicar_knn(k, dados_normalizados, instancia_teste, 'euclidean')
        print(resultado,"\tk = ", k)

    print('\n### RESULTADOS COM DISTANCIA DE CHEBYSHEV ###\n')
    for k in range(1, 101, 2):
        resultado = aplicar_knn(k, dados_normalizados, instancia_teste, 'chebyshev')
        print(resultado,"\tk = ", k)

    print('\n### RESULTADOS COM DISTANCIA MANHATTAN ###\n')
    for k in range(1, 101, 2):
        resultado = aplicar_knn(k, dados_normalizados, instancia_teste, 'manhattan')
        print(resultado,"\tk = ", k)


main()
