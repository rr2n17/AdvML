from builtins import zip
from builtins import map
from builtins import range
from .rakeld import RakelD
import copy
import numpy as np
import random
from scipy import sparse


class RakelO(RakelD):
    """

    Overlapping RAndom k-labELsets multi-label classifier.

    """

    def __init__(self, classifier=None, model_count=None, labelset_size=None, require_dense=None):
        super(RakelO, self).__init__(
            classifier=classifier, require_dense=require_dense)
        self.model_count = int(model_count)
        self.labelset_size = labelset_size
        self.copyable_attrs = ['model_count',
                               'labelset_size', 'require_dense', 'classifier']

    def generate_partition(self, X, y):
        """Randomly divide the label space

        This function randomly divides the label space of `n_labels` into `model_count`  
        equal subspaces of size `labelset_size`.  

        :param X: not used, maintained for api compatibility
        :param y: binary indicator matrix with label assignments
        :type y: dense or sparse matrix of {0, 1} (n_samples, n_labels)

        Sets `self.partition`, `self.label_count`.

        """
        label_sets = []
        self.label_count = y.shape[1]
        free_labels = range(self.label_count)

        while len(label_sets) < self.model_count:
            label_set = random.sample(free_labels, self.labelset_size)
            if label_set not in label_sets:
                label_sets.append(label_set)

        self.partition = label_sets
        assert len(self.partition) == self.model_count

    def predict(self, X):
        """Predict probabilities of label assignments for X

        Internally this method uses a sparse CSC representation for X 
        (:py:class:`scipy.sparse.csr_matrix`).

        :param X: input features
        :type X: dense or sparse matrix (n_samples, n_labels)
        :returns: matrix with label assignment probabilities
        :rtype: sparse matrix of float (n_samples, n_labels)
        
        """
        predictions = [
            self.ensure_input_format(self.ensure_input_format(
                c.predict(X)), sparse_format='csc', enforce_sparse=True)
            for c in self.classifiers
        ]

        votes = sparse.csc_matrix(
            (predictions[0].shape[0], self.label_count), dtype='int')
        for model in range(self.model_count):
            for label in range(len(self.partition[model])):
                votes[:, self.partition[model][label]] = votes[
                    :, self.partition[model][label]] + predictions[model][:, label]

        voters = list(map(float, votes.sum(axis=0).tolist()[0]))

        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            votes[row, column] = np.round(
                votes[row, column] / float(voters[column]))

        return self.ensure_input_format(votes, enforce_sparse=False)
