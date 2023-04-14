import numpy as np
import math


class gaussNaiveBayes:
    def separate_by_classes(self):
        ''' This function separates our dataset in subdatasets by classes '''
        classes_index = {}
        self.subdatasets = {}
        self.classes, counts = np.unique(self.labels, return_counts=True)
        self.class_freq = dict(
            zip(self.classes, counts / self.labels.shape[0]))
        for class_type in self.classes:
            classes_index[class_type] = np.argwhere(self.labels == class_type)
            self.subdatasets[class_type] = self.data[classes_index[class_type], :][:, 0]

    def calculate_means_and_std(self):
        ''' Calculates means and standard deviations (call separate_by_classes first)'''
        self.means = {}
        self.std = {}
        for class_type in self.classes:
            self.means[class_type] = np.mean(
                self.subdatasets[class_type], axis=0)
            self.std[class_type] = np.std(self.subdatasets[class_type], axis=0)

    def choose_prediction(self):
        max_val = -np.inf
        for cls in self.classes:
            value = self.predictions[cls]
            if value > max_val:
                self.prediction = cls
                max_val = value
        return self.prediction

    def calculate_predictions(self, input: np.ndarray):
        alpha = 0.0123
        std = np.array([self.std[cls] for cls in self.classes])
        means = np.array([self.means[cls] for cls in self.classes])
        class_freq_log = np.log(
            np.array([self.class_freq[cls] for cls in self.classes]))

        exponent = -(((input - means) ** 2) / (2 * (std ** 2 + alpha)))
        log_prob = np.sum(exponent - np.log(np.sqrt(2 * math.pi *
                                                    (std ** 2 + alpha))), axis=1) + class_freq_log

        self.predictions = dict(zip(self.classes, log_prob))

    def predict(self, input: np.ndarray, labels: np.ndarray):
        ''' This function predicts the class for a new input '''
        float_input = input
        predictions = []
        for i in range(float_input.shape[0]):
            self.calculate_predictions(float_input[i])
            predictions.append(self.choose_prediction())
        self.accuracy = np.mean(predictions == labels)
        return np.array(predictions)

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels
        self.separate_by_classes()
        self.calculate_means_and_std()
