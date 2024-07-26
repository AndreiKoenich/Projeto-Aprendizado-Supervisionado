import utils
from sklearn import tree as tree
#import matplotlib.pyplot as plt

'''
def _print_tree(func):
    def x(train_data_x, train_data_y):
        (model, accuracy) = func(train_data_x, train_data_y)
        plt.figure(figsize=(20,10))
        tree.plot_tree(
                model,
                filled=True,
                feature_names=train_data_x.columns,
                class_names=model.classes_)
        plt.show()
        return (model, accuracy)
    x.__name__ = func.__name__
    return x
'''

def train_decision_tree(train_data_x, train_data_y, test_data_x, test_data_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_data_x, train_data_y)
    accuracy = model.score(test_data_x, test_data_y)
    return (model, accuracy)