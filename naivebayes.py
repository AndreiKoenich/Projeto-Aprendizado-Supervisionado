from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np
from utils import cria_instancia_teste, ler_dados_csv, normalizar_dados, one_hot_encoding
import pandas as pd


def combinar_naive_bayes(dados, instancia_teste):
    # Separar atributos numéricos e categóricos
    atributos_numericos = ['Temperature','Humidity','Wind Speed','Precipitation (%)','Atmospheric Pressure','UV Index','Visibility (km)']
    atributos_categoricos = ['Cloud Cover_clear','Cloud Cover_cloudy','Cloud Cover_overcast','Cloud Cover_partly cloudy','Season_Autumn','Season_Spring','Season_Winter','Season_Summer','Location_coastal','Location_inland','Location_mountain']

    # Separar os dados
    X_numerico = dados[atributos_numericos].astype(float)
    X_categorico = dados[atributos_categoricos]
    y = dados['Weather Type']
    
    # Dividir instância de teste
    instancia_numerico = instancia_teste[atributos_numericos].astype(float)
    instancia_categorico = instancia_teste[atributos_categoricos]

    # Criar e treinar modelos
    modelo_gaussiano = GaussianNB()
    modelo_multinomial = MultinomialNB()
    
    modelo_gaussiano.fit(X_numerico, y)
    modelo_multinomial.fit(X_categorico, y)
    
    # Calcular probabilidades para a instância de teste
    prob_numerico = modelo_gaussiano.predict_proba(instancia_numerico)
    prob_categorico = modelo_multinomial.predict_proba(instancia_categorico)

    # Computa as classes
    classes = modelo_gaussiano.classes_

    # Inicializa os contadores para calcular a acuracia
    correct = 0
    total = 0

    for row_index in range(len(instancia_teste)):
        # Estimar a classe
        probabilidade_combinada = prob_numerico[row_index, :] * prob_categorico[row_index, :]
        classe = classes[np.argmax(probabilidade_combinada)]

        # Checar se a estimativa foi correta
        if classe == list(instancia_teste['Weather Type'])[row_index]:
            correct += 1
        total += 1

    # Calcular a acuracia
    return float(correct) / float(total)


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
        results = combinar_naive_bayes(training_data, test_data)
        print(results)


main()
