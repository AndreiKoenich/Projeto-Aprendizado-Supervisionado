import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def log_usage(func):
    def x(*args, **kwargs):
        print('Using function', func.__name__)
        return func(*args, **kwargs)
    x.__name__ = func.__name__
    return x


# IO/DATA
# @log_usage
def import_data():
    filename = 'weather_classification_data.csv'
    data = ler_dados_csv(filename)
    data = one_hot_encoding(data)
    data = normalizar_dados(data)
    return data


def scramble_data(data):
    return data.sample(frac=1).reset_index(drop=True)


def train_test_split(data, *, test_data_percent=0.15):
    test_data_count = int(test_data_percent * len(data))
    train_data = data[test_data_count:]
    test_data = data[:test_data_count]
    return (train_data, test_data)


def xy_split(data, *, columns):
    x = data.drop(columns=columns)
    y = data[columns]
    if len(columns) == 1:
        y = y[columns[0]]
    return (x, y)

def _one_hot_encoding(data, *, columns):
    data = pd.get_dummies(data, columns=columns)

    for column in data.columns:
        if data[column].dtype == bool:
            data[column] = data[column].astype(int)

    return data




# INPUT FUNCTIONS
atributos_numericos = ['Temperature','Humidity','Wind Speed','Precipitation (%)','Atmospheric Pressure','UV Index','Visibility (km)']
atributos_categoricos = ['Cloud Cover_clear','Cloud Cover_cloudy','Cloud Cover_overcast','Cloud Cover_partly cloudy','Season_Autumn','Season_Spring','Season_Winter','Season_Summer','Location_coastal','Location_inland','Location_mountain']

def identity(x):
    return x

def extract_numerical(x):
    return x[atributos_numericos]

def extract_categorical(x):
    return x[atributos_categoricos]

def importar_dados():
    pass

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

def cross_validation(n, k):
    prev_i = 0
    for i in range(1, k):
        yield (prev_i, i*(n // k))
        prev_i = i*(n // k)
    yield (prev_i, n)

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


def evaluate_model(model, test_data):
    # Split test data
    target = test_data['Weather Type']
    data = test_data.drop(columns=['Weather Type'])

    # Predict results
    prediction = model.predict(data)

    # Calculate accuracy
    accuracy = float((prediction == target).sum()) / float(len(prediction))

    return accuracy

