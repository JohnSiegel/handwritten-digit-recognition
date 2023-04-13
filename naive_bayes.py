import numpy as np


class gaussNaiveBayes:
    def separate_by_classes(self, data: np.ndarray, labels: np.ndarray):
        ''' This function separates our dataset in subdatasets by classes '''
        classes_index = {}
        subdatasets = {}
        self.classes, counts = np.unique(labels, return_counts=True)
        self.class_freq = dict(
            zip(self.classes, counts / labels.shape[0]))
        for class_type in self.classes:
            classes_index[class_type] = np.argwhere(labels == class_type)
            subdatasets[class_type] = data[classes_index[class_type], :][:, 0]
        return subdatasets
