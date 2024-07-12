from sklearn import tree
import matplotlib.pyplot as plt
from utils import cria_instancia_teste, ler_dados_csv, normalizar_dados, one_hot_encoding


def aplicar_arvore_decisao(dados_normalizados, instancia_teste):
    # Separar atributo alvo
    atributo_alvo = dados_normalizados['Weather Type']
    
    # Criar modelo
    clf = tree.DecisionTreeClassifier()
    
    # Treinar o modelo com os dados normalizados
    clf.fit(dados_normalizados.drop(columns=['Weather Type']), atributo_alvo)
    
    # Plotar a árvore de decisão
    #plt.figure(figsize=(20,10))
    #tree.plot_tree(clf, filled=True, feature_names=dados_normalizados.columns[:-1], class_names=clf.classes_)
    #plt.show()
    
    # Fazer a predição para a instância de teste
    predicao = clf.predict(instancia_teste.drop(columns=['Weather Type']))
    return predicao

def main():
    nome_arquivo = 'weather_classification_data.csv'
    dados = ler_dados_csv(nome_arquivo)
    dados_encoded = one_hot_encoding(dados)
    dados_normalizados = normalizar_dados(dados_encoded)

    instancia_teste = cria_instancia_teste()

    predicao = aplicar_arvore_decisao(dados_normalizados, instancia_teste)
    print(f'Predição para a instância de teste: {predicao}')

main()
