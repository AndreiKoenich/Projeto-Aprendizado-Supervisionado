import csv
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
import numpy as np

def ler_dados_csv(nome_arquivo):
    dados = []
    with open(nome_arquivo, mode='r', encoding='utf-8') as arquivo_csv:
        leitor_csv = csv.DictReader(arquivo_csv)
        for linha in leitor_csv:
            dados.append(linha)
    return dados

def one_hot_encoding(dados):
    df = pandas.DataFrame(dados)
    df_encoded = pandas.get_dummies(df, columns=['Cloud Cover','Season','Location'])

    for column in df_encoded.columns:
        if df_encoded[column].dtype == bool:
            df_encoded[column] = df_encoded[column].astype(int)

    return df_encoded

def normalizar_dados(dados_encoded):
    # Extraindo o atributo alvo 'Weather Type'
    weather_type = dados_encoded['Weather Type']
    dados_para_normalizar = dados_encoded.drop(columns=['Weather Type'])
    
    # Aplicando a normalização nos dados, exceto 'Weather Type'
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados_para_normalizar.astype(float))
    
    # Recriando o DataFrame com os dados normalizados
    colunas_normalizadas = dados_para_normalizar.columns
    dados_normalizados_df = pandas.DataFrame(dados_normalizados, columns=colunas_normalizadas)
    
    # Adicionando de volta o atributo 'Weather Type'
    dados_normalizados_df['Weather Type'] = weather_type.reset_index(drop=True)
    
    return dados_normalizados_df

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

def cria_instancia_teste():
    dados_exemplo = {
        'Temperature': [0.410448],
        'Humidity': [0.494382],
        'Wind Speed': [0.144330],
        'Precipitation (%)': [0.146789],
        'Atmospheric Pressure': [0.547746],
        'UV Index': [0.357143],
        'Visibility (km)': [0.275],
        'Cloud Cover_clear': [1.0],
        'Cloud Cover_cloudy': [0.0],
        'Cloud Cover_overcast': [0.0],
        'Cloud Cover_partly cloudy': [0.0],
        'Season_Autumn': [0.0],
        'Season_Spring': [1.0],
        'Season_Summer': [0.0],
        'Season_Winter': [0.0],
        'Location_coastal': [0.0],
        'Location_inland': [0.0],
        'Location_mountain': [1.0],
        'Weather Type': ['None']
    }
    instancia_teste = pandas.DataFrame(dados_exemplo)
    return instancia_teste

def main():
    nome_arquivo = 'weather_classification_data.csv'
    dados = ler_dados_csv(nome_arquivo)
    dados_encoded = one_hot_encoding(dados)
    dados_normalizados = normalizar_dados(dados_encoded)

    #print(dados)
    #print(dados_encoded)
    print(dados_normalizados)

    instancia_teste = cria_instancia_teste()
    resultado = combinar_naive_bayes(dados_normalizados, instancia_teste)

    print(resultado)

main()
