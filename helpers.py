import numpy as np


def normalize_dataset(data: np.ndarray):
    '''
    normalize dataset to have values in the range [0, 1]

    @return normalized dataset
    '''
    casted_data = data.astype(float)
    return casted_data / np.max(casted_data)
