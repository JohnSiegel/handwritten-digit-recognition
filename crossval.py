"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
from typing import Callable, Dict, Tuple
import numpy as np


def cross_validate(trainer: Callable[[np.ndarray, np.ndarray, Dict], object],
                   predictor: Callable[[np.ndarray, object], np.ndarray],
                   all_data: np.ndarray,
                   all_labels: np.ndarray,
                   folds: int,
                   params: Dict) -> Tuple[float, list]:
    """Perform cross validation with random splits.
    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    accuracy_scores = np.zeros(folds)
    num_examples = all_data.shape[1]
    indices = np.array(range(num_examples), dtype=int)
    num_examples_per_fold = np.ceil(num_examples / folds).astype(int)
    num_examples_to_add = num_examples_per_fold * folds - num_examples
    indices = np.append(indices, -np.ones(num_examples_to_add, dtype=int))
    indices = indices.reshape((num_examples_per_fold, folds))
    models = []
    shuffle_indices = np.random.permutation(range(num_examples))
    shuffled_data = all_data[:, shuffle_indices]
    shuffled_labels = all_labels[shuffle_indices]
    for fold_index in range(folds):
        validation_indices = indices[:,
                                     fold_index][indices[:, fold_index] != -1]
        training_indices = np.delete(
            np.arange(num_examples), validation_indices)
        training_data = shuffled_data[:, training_indices]
        training_labels = shuffled_labels[training_indices]
        validation_data = shuffled_data[:, validation_indices]
        validation_labels = shuffled_labels[validation_indices]
        models.append(trainer(training_data, training_labels, params))
        predicted_labels = predictor(validation_data, models[-1])
        correct_labels = predicted_labels == validation_labels
        accuracy_score = correct_labels.sum() / validation_labels.shape[0]
        accuracy_scores[fold_index] = accuracy_score
    mean_accuracy = np.mean(accuracy_scores)
    return mean_accuracy, models
