import csv
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def ler_dados_csv(nome_arquivo):
    dados = []
    with open(nome_arquivo, mode='r', encoding='utf-8') as arquivo_csv:
        leitor_csv = csv.DictReader(arquivo_csv)
        for linha in leitor_csv:
            dados.append(linha)
    return dados

def one_hot_encoding(dados):
    df = pandas.DataFrame(dados)
    df_encoded = pandas.get_dummies(df, columns=['Cloud Cover', 'Season', 'Location'])

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

def aplicar_knn(k, dados_normalizados, instancia_teste):

    atributo_alvo = dados_normalizados['Weather Type']

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(dados_normalizados, atributo_alvo)
    resultado = knn.predict(instancia_teste)
    return resultado

def main():
    nome_arquivo = 'weather_classification_data.csv'
    dados = ler_dados_csv(nome_arquivo)
    dados_encoded = one_hot_encoding(dados)
    dados_normalizados = normalizar_dados(dados_encoded)

    #print(dados)
    #print(dados_encoded)
    print(dados_normalizados)

main()
