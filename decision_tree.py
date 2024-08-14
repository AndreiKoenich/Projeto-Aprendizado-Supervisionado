import utils
from sklearn import tree as tree

def train_decision_tree(train_data_x, train_data_y, test_data_x, test_data_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_data_x, train_data_y)
    accuracy = model.score(test_data_x, test_data_y)
    utils.makeRocCurve(model, 'train_decision_tree', test_data_x, test_data_y,  train_data_y)
    return (model, accuracy)
