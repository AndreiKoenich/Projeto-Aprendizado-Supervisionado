from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np
from utils import cria_instancia_teste, ler_dados_csv, normalizar_dados, one_hot_encoding

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

    # Multiplicar as probabilidades
    prob_combinada = prob_numerico * prob_categorico
    classes = modelo_gaussiano.classes_
    resultado = classes[np.argmax(prob_combinada)]
    
    return resultado

def main():
    nome_arquivo = 'weather_classification_data.csv'
    dados = ler_dados_csv(nome_arquivo)
    dados_encoded = one_hot_encoding(dados)
    dados_normalizados = normalizar_dados(dados_encoded)

    #print(dados)
    #print(dados_encoded)
    #print(dados_normalizados)

    instancia_teste = cria_instancia_teste()
    resultado = combinar_naive_bayes(dados_normalizados, instancia_teste)

    print(resultado)

main()
