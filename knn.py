from sklearn.neighbors import KNeighborsClassifier
import utils


def _train_knn(train_x, train_y, test_x, test_y, *, metric, min_k=1, max_k=11, k_step=2):
    for k in range(min_k, max_k, k_step):
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(train_x, train_y)
        accuracy = model.score(test_x, test_y)
        yield (model, accuracy)


# Uncomment decorators to activate extra functionality
# @utils.log_usage
def train_knn_euc(train_x, train_y, test_x, test_y, *, min_k=1, max_k=11, k_step=2):
    model_generator = _train_knn(
            train_x, train_y, test_x, test_y,
            min_k=min_k, max_k=max_k, k_step=k_step,
            metric="euclidean")
    (best_model, best_accuracy) = max(model_generator, key=lambda x: x[1])
    return (best_model, best_accuracy)


# Uncomment decorators to activate extra functionality
# @utils.log_usage
def train_knn_che(train_x, train_y, test_x, test_y, *, min_k=1, max_k=11, k_step=2):
    model_generator = _train_knn(
            train_x, train_y, test_x, test_y,
            min_k=min_k, max_k=max_k, k_step=k_step,
            metric="chebyshev")
    (best_model, best_accuracy) = max(model_generator, key=lambda x: x[1])
    return (best_model, best_accuracy)


# Uncomment decorators to activate extra functionality
# @utils.log_usage
def train_knn_man(train_x, train_y, test_x, test_y, *, min_k=1, max_k=11, k_step=2):
    model_generator = _train_knn(
            train_x, train_y, test_x, test_y,
            min_k=min_k, max_k=max_k, k_step=k_step,
            metric="manhattan")
    (best_model, best_accuracy) = max(model_generator, key=lambda x: x[1])
    return (best_model, best_accuracy)


def main():
    data = utils.import_data()
    data = utils.scramble_data(data)

    train_data, test_data = utils.train_test_split(data)
    train_x, train_y = utils.xy_split(train_data, columns=['Weather Type'])
    test_x, test_y = utils.xy_split(test_data, columns=['Weather Type'])

    (model, accuracy) = train_knn_euc(train_x, train_y, test_x, test_y)
    print(f"Model accuracy [knn euclidean, k={model.n_neighbors}] = {accuracy}")
    (model, accuracy) = train_knn_che(train_x, train_y, test_x, test_y)
    print(f"Model accuracy [knn chebyshev, k={model.n_neighbors}] = {accuracy}")
    (model, accuracy) = train_knn_man(train_x, train_y, test_x, test_y)
    print(f"Model accuracy [knn manhattan, k={model.n_neighbors}] = {accuracy}")


if __name__ == '__main__':
    main()


