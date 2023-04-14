import numpy as np


def one_hot_encode(labels: np.ndarray):
    num_classes = np.unique(labels).shape[0]
    one_hot = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        np.put(one_hot[i], labels[i], 1)
    return one_hot


def preprocess_data(data: np.ndarray):
    reshaped = np.zeros((data.shape[0], 1, 28, 28))
    for i in range(data.shape[0]):
        temp = data[i, :]
        temp = np.ravel(temp)
        temp = temp.reshape(28, 28)
        reshaped[i, 0, :, :] = temp
    return reshaped
