from csv import reader
from math import sqrt
import operator


def log_data_set(training_data: str, test_data: str)->tuple():
    training_feature_vector: list = []
    test_feature_vector: list = []
    with open(training_data, 'r', encoding='utf8') as file_open:
        lines = reader(file_open)
        dataset: list = list(lines)
        for x in range(len(dataset)):
            for y in range(len(range(3))):
                dataset[x][y] = float(dataset[x][y])
            training_feature_vector.append(dataset[x])

    with open(test_data, 'r', encoding='utf8') as file_open:
        lines = reader(file_open)
        dataset: list = list(lines)
        for x in range(len(dataset)):
            for y in range(len(range(3))):
                dataset[x][y] = float(dataset[x][y])
            test_feature_vector.append(dataset[x])

    return tuple([training_feature_vector, test_feature_vector[0]])


def euclidian_distance(test_variable: list, training_variable: list)->float:
    distance: float = 0
    for x in range(len(test_variable)):
        distance += pow((test_variable[x]-training_variable[x]), 2)
    return sqrt(distance)


def k_nearest_neighbors(test_data: list, training_data: list, k: int)->list:
    distance: list = []
    for i in range(len(training_data)):
        dist = euclidian_distance(test_data, training_data[i])
        distance.append((training_data[i], dist))

    distance.sort(key=operator.itemgetter(1))
    neighbors: list = []
    for i in range(k):
        neighbors.append(distance[i][0])
    return neighbors


def response_neighboars(neighbors: list)->str:
    global sorted_vote
    all_neighbors: dict = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]  # Dernier élément de la liste (couleur)
        if response in all_neighbors:
            all_neighbors[response] += 1
        else:
            all_neighbors.setdefault(response, 1)
        sorted_vote = sorted(all_neighbors.items(),
                             key=operator.itemgetter(1), reverse=True)
    return sorted_vote[0][0]


def main(training_data: str, test_data: str, k: int)->str:
    training_feature_vector, test_feature_vector = log_data_set(training_data,
                                                                test_data)
    classifier_prediction = []  # predictions
    for x in range(len(test_feature_vector)):
        neighbors = k_nearest_neighbors(test_feature_vector,
                                        training_feature_vector, k)
        result = response_neighboars(neighbors)
        classifier_prediction.append(result)
    return classifier_prediction[0]
