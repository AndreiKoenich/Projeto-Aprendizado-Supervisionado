import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def ler_dados_csv(nome_arquivo):
    dados = []
    with open(nome_arquivo, mode='r', encoding='utf-8') as arquivo_csv:
        leitor_csv = csv.DictReader(arquivo_csv)
        for linha in leitor_csv:
            dados.append(linha)
    return dados

def one_hot_encoding(dados):
    df = pd.DataFrame(dados)
    df_encoded = pd.get_dummies(df, columns=['Cloud Cover', 'Season', 'Location'])

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
    dados_normalizados_df = pd.DataFrame(dados_normalizados, columns=colunas_normalizadas)
    
    # Adicionando de volta o atributo 'Weather Type'
    dados_normalizados_df['Weather Type'] = weather_type.reset_index(drop=True)
    
    return dados_normalizados_df

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
    instancia_teste = pd.DataFrame(dados_exemplo)
    return instancia_teste

def cross_validation(n, k):
    prev_i = 0
    for i in range(1, k):
        yield (prev_i, i*(n // k))
        prev_i = i*(n // k)
    yield (prev_i, n)
