import numpy as np
from naive_bayes import gaussNaiveBayes


def test_separate_by_classes():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    labels = np.array([0, 1, 2, 1, 0])
    gnb = gaussNaiveBayes()
    subdatasets = gnb.separate_by_classes(data, labels)
    assert np.allclose(subdatasets[0], np.array([[1, 2], [9, 10]]))
    assert np.allclose(subdatasets[1], np.array([[3, 4], [7, 8]]))
    assert np.allclose(subdatasets[2], np.array([[5, 6]]))
    assert gnb.classes.tolist() == [0, 1, 2]
    assert np.allclose(list(gnb.class_freq.values()), [0.4, 0.4, 0.2])

    data = np.array([[1, 2, 10, 20, 40], [3, 4, 1, 2, 5], [5, 6, 200, 128, 1], [
                    7, 8, 1, 2, 3], [100, 100, 100, 9, 10]])
    labels = np.array([0, 2, 0, 2, 1])
    gnb = gaussNaiveBayes()
    subdatasets = gnb.separate_by_classes(data, labels)
    assert np.allclose(subdatasets[0], np.array(
        [[1, 2, 10, 20, 40], [5, 6, 200, 128, 1]]))
    assert np.allclose(subdatasets[1], np.array([[100, 100, 100, 9, 10]]))
    assert np.allclose(subdatasets[2], np.array([[3, 4, 1, 2, 5], [
        7, 8, 1, 2, 3]]))
    assert gnb.classes.tolist() == [0, 1, 2]
    assert np.allclose(list(gnb.class_freq.values()), [0.4, 0.2, 0.4])


test_separate_by_classes()
