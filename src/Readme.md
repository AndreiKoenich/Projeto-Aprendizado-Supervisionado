UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL - Agosto de 2024
Aprendizado de Máquina
Trabalho 2 - Avaliação de Modelos

ANDREI POCHMANN KOENICH
BRUNO FERREIRA AIRES
FELIPE KAISER SCHNITZLER
FELIPE SOUZA DIDIO

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

DESCRIÇÕES DOS ARQUIVOS:

weather_classification_data.csv 		- Contém os dados a serem lidos e analisados pelos modelos.
naivebayes.py							- Contém as funções referentes à aplicação do algoritmo Naive Bayes.
knn.py 									- Contém as funções referentes à aplicação do algoritmo k-Nearest Neighbors.
decision.tree.py 						- Contém as funções referentes à aplicação do algoritmo de Árvore de Decisão.
utils.py								- Contém funções para leitura dos dados de entrada, pré-processamento dos dados, separação dos dados em treinamento, validação e teste e avaliação dos modelos.
main.py									- Contém as chamadas das funções existentes nos demais arquivos, para obtenção dos valores de acurácia obtidos para cada um dos cinco modelos nos quatro testes descritos no relatório, além das curvas PR e ROC.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

INSTRUÇÕES:

Deve ser executado o arquivo "main.py", presente na pasta "src" (os integrantes do grupo executam o arquivo com o comando "py -3 main.py").
Depois disso, serão executados os quatro testes possíveis para cada um dos cinco modelos, conforme explicado no relatório anexado.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

IMPORTAÇÕES UTILIZADAS:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, label_binarize
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree as tree
import warnings
import pprint
import shutil
import os
import random
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, average_precision_score, precision_recall_curve
from collections import Counter
import matplotlib.pyplot as plt

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

DESCRIÇÃO DAS SAÍDAS GERADAS:

No mesmo diretório do arquivo "main.py", será gerado o diretório "tests", contendo os diretórios "test_1", "test_2", "test_3" e "test_4".
Cada um desses diretórios possui as pastas "decision_tree", "kn_che", "knn_euc", "knn_man" e "naive_bayes" e o arquivo texto "test_data.txt".

O diretório "test_1" contém os resultados obtidos para os testes COM A PRESENÇA DE OUTLIERS e utilização de HOLDOUT, com cada subdiretório contendo as curvas PR e ROC geradas para cada um dos cinco modelos.
O diretório "test_2" contém os resultados obtidos para os testes COM A PRESENÇA DE OUTLIERS e utilização de BOOTSTRAP, com cada subdiretório contendo as curvas PR e ROC geradas para cada um dos cinco modelos.
O diretório "test_3" contém os resultados obtidos para os testes SEM A PRESENÇA DE OUTLIERS e utilização de HOLDOUT, com cada subdiretório contendo as curvas PR e ROC geradas para cada um dos cinco modelos.
O diretório "test_4" contém os resultados obtidos para os testes SEM A PRESENÇA DE OUTLIERS e utilização de BOOTSTRAP, com cada subdiretório contendo as curvas PR e ROC geradas para cada um dos cinco modelos.

O arquivo texto "test_data.txt" presente em cada um dos quatro diretórios criados informa o valor do hiperparâmetro otimizado para cada um dos modelos (exceto Naive Bayes),
além do valor de acurácia obtido para cada um dos modelos.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------





