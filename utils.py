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
    data = remove_outliers(data)
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
    dados = pd.read_csv(nome_arquivo)
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

def remove_outliers(dados):
    # Colunas do dataset que serão analisadas
    columns = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']

    for column in columns:
        # Obtem os dois "quadrantes" que seram considerados para o cálculo
        Q1 = dados[column].quantile(0.25)
        Q3 = dados[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define os limites com base no IQR calculado
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove as linhas que contém um outlier no atributo.
        dados = dados[(dados[column] >= lower_bound) & (dados[column] <= upper_bound)]

    return dados

def cross_validation(n, k):
    prev_i = 0
    for i in range(1, k):
        yield (prev_i, i*(n // k))
        prev_i = i*(n // k)
    yield (prev_i, n)


def evaluate_model(model, test_data):
    # Split test data
    target = test_data['Weather Type']
    data = test_data.drop(columns=['Weather Type'])

    # Predict results
    prediction = model.predict(data)

    # Calculate accuracy
    accuracy = float((prediction == target).sum()) / float(len(prediction))

    return accuracy

