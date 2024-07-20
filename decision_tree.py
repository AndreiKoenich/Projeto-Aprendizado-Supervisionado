import utils
import matplotlib.pyplot as plt
from sklearn import tree as tree


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


# Uncomment decorators to activate extra functionality
# @utils.log_usage
# @_print_tree
def train_decision_tree(train_data_x, train_data_y, test_data_x, test_data_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_data_x, train_data_y)
    accuracy = model.score(test_data_x, test_data_y)
    return (model, accuracy)


def main():
    data = utils.import_data()
    data = utils.scramble_data(data)
    
    train_data, test_data = utils.train_test_split(data)
    train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
    test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])

    (model, accuracy) = train_decision_tree(train_x, train_y, test_x, test_y)
    print(f"Model accuracy [decision tree] = {accuracy}")


if __name__ == '__main__':
    main()

