import csv
import pandas
from sklearn.preprocessing import MinMaxScaler

def ler_dados_csv(nome_arquivo):
    dados = []
    with open(nome_arquivo, mode='r', encoding='utf-8') as arquivo_csv:
        leitor_csv = csv.DictReader(arquivo_csv)
        for linha in leitor_csv:
            dados.append(linha)
    return dados

def one_hot_encoding(dados):
    df = pandas.DataFrame(dados)
    df_encoded = pandas.get_dummies(df, columns=['Cloud Cover', 'Season', 'Location', 'Weather Type'])

    for column in df_encoded.columns:
        if df_encoded[column].dtype == bool:
            df_encoded[column] = df_encoded[column].astype(int)

    return df_encoded

def normalizar_dados(dados_encoded):
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados_encoded.astype(float))
    return pandas.DataFrame(dados_normalizados, columns=dados_encoded.columns)

def main():
    nome_arquivo = 'weather_classification_data.csv'
    dados = ler_dados_csv(nome_arquivo)
    dados_encoded = one_hot_encoding(dados)
    dados_normalizados = normalizar_dados(dados_encoded)

    #print(dados)
    #print(dados_encoded)
    print(dados_normalizados)

main()
