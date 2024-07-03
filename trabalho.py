import csv

def ler_dados_csv(nome_arquivo):
    dados = []
    with open(nome_arquivo, mode='r', encoding='utf-8') as arquivo_csv:
        leitor_csv = csv.DictReader(arquivo_csv)
        for linha in leitor_csv:
            dados.append(linha)
    return dados

# Exemplo de uso:
nome_arquivo = 'weather_classification_data.csv'
dados = ler_dados_csv(nome_arquivo)
for instancia in dados:
    print(instancia)
