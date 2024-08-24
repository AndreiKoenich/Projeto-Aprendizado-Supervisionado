import utils
from sklearn import tree as tree


def _dt_generator(train_x, train_y, test_x, test_y, *, min_depth=1, max_depth=100, step=1):
    for depth in range(min_depth, max_depth, step):
        model = tree.DecisionTreeClassifier(max_depth=depth)
        model.fit(train_x, train_y)
        accuracy = model.score(test_x, test_y)
        yield (model, accuracy)


def train_decision_tree(train_data_x, train_data_y, test_data_x, test_data_y):
    model_generator = _dt_generator(
            train_data_x, train_data_y, test_data_x, test_data_y,
            min_depth=1, max_depth=100, step=1)
    (best_model, best_accuracy) = max(model_generator, key=lambda x: x[1])
    return (best_model, best_accuracy)

