import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
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
    #data = remove_outliers(data)
    data = one_hot_encoding(data)
    data = normalizar_dados(data)
    return data

def scramble_data(data):
    return data.sample(frac=1).reset_index(drop=True)

# Função para dividir o conjunto de dados manualmente
def split_data_manual(data):
    n = len(data)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def stratified_split(data):

    # Filtrar instâncias com base no valor do atributo 'Weather Type'
    conjunto_snowy = data[data['Weather Type'] == 'Snowy']
    conjunto_rainy = data[data['Weather Type'] == 'Rainy']
    conjunto_sunny = data[data['Weather Type'] == 'Sunny']
    conjunto_cloudy = data[data['Weather Type'] == 'Cloudy']

    # Dividindo manualmente cada um dos conjuntos conjunto_snowy, conjunto_rainy, conjunto_sunny e conjunto_cloudy
    train_conjunto_snowy, val_conjunto_snowy, test_conjunto_snowy = split_data_manual(conjunto_snowy)
    train_conjunto_rainy, val_conjunto_rainy, test_conjunto_rainy = split_data_manual(conjunto_rainy)
    train_conjunto_sunny, val_conjunto_sunny, test_conjunto_sunny = split_data_manual(conjunto_sunny)
    train_conjunto_cloudy, val_conjunto_cloudy, test_conjunto_cloudy = split_data_manual(conjunto_cloudy)

    # Concatenar os conjuntos de treinamento, validação e teste
    train_data = pd.concat([train_conjunto_snowy, train_conjunto_rainy, train_conjunto_sunny, train_conjunto_cloudy])
    val_data = pd.concat([val_conjunto_snowy, val_conjunto_rainy, val_conjunto_sunny, val_conjunto_cloudy])
    test_data = pd.concat([test_conjunto_snowy, test_conjunto_rainy, test_conjunto_sunny, test_conjunto_cloudy])

    # Opcional: Salvar cada conjunto em um arquivo .csv
    #train_data.to_csv('train_data.csv', index=False)
    #val_data.to_csv('val_data.csv', index=False)
    #test_data.to_csv('test_data.csv', index=False)

    return (train_data, val_data, test_data)


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

def evaluate_model(model, test_data):
    # Split test data
    target = test_data['Weather Type']
    data = test_data.drop(columns=['Weather Type'])

    # Predict results
    prediction = model.predict(data)

    # Calculate accuracy
    accuracy = float((prediction == target).sum()) / float(len(prediction))

    return accuracy

def makeRocCurve(model, model_name, test_data_x, test_data_y, train_data_y):
    label_binarizer = LabelBinarizer().fit(train_data_y)
    y_onehot_test = label_binarizer.transform(test_data_y)
    y_onehot_test.shape  # (n_samples, n_classes)

    classes = ["Cloudy", "Sunny", "Snowy", "Rainy"]
    for class_of_interest in label_binarizer.classes_:
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
        rest = [c for c in classes if c != class_of_interest]

        y_score = model.predict_proba(test_data_x)
        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"One-vs-Rest ROC curves:\n{class_of_interest} vs ({', '.join(rest)})",
        )

        display.plot
        plt.savefig(f'roc/{class_of_interest}-{model_name}.png')


def writeFile(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)