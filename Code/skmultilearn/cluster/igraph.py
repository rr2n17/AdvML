from __future__ import absolute_import
from builtins import range
from .base import LabelCooccurenceClustererBase
import numpy as np
import igraph as ig


class IGraphLabelCooccurenceClusterer(LabelCooccurenceClustererBase):

    """Clusters the label space using igraph community detection methods

    Parameters
    ----------

    method : enum from `IGraphLabelCooccurenceClusterer.METHODS`
        the igraph community detection method that will be used

    weighted: boolean
            Decide whether to generate a weighted or unweighted graph.

    include_self_edges : boolean
            Decide whether to include self-edge i.e. label 1 - label 1 in co-occurrence graph

    """

    METHODS = {
        'fastgreedy': lambda graph, w = None: graph.community_fastgreedy(weights=w).as_clustering(),
        'infomap': lambda graph, w = None: graph.community_infomap(edge_weights=w),
        'label_propagation': lambda graph, w = None: graph.community_label_propagation(weights=w),
        'leading_eigenvector': lambda graph, w = None: graph.community_leading_eigenvector(weights=w),
        'multilevel': lambda graph, w = None: graph.community_multilevel(weights=w),
        'walktrap': lambda graph, w = None: graph.community_walktrap(weights=w).as_clustering(),
    }

    def __init__(self, method=None, weighted=None, include_self_edges=None):
        super(IGraphLabelCooccurenceClusterer, self).__init__(
            weighted=weighted, include_self_edges=include_self_edges)
        self.method = method

        if method not in IGraphLabelCooccurenceClusterer.METHODS:
            raise ValueError(
                "{} not a supported igraph community detection method".format(method))

    def fit_predict(self, X, y):
        """Performs clustering on y and returns list of label lists

        Builds a label coocurence_graph using :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix` on `y` and then detects communities using a selected `method`.

        Parameters
        ----------
        X : sparse matrix (n_samples, n_features), feature space, not used in this clusterer
        y : sparse matrix (n_samples, n_labels), label space

        Returns
        -------
        partition: list of lists : list of lists label indexes, each sublist represents labels that are in that community


        """
        self.generate_coocurence_adjacency_matrix(y)

        if self.is_weighted:
            self.weights = dict(weight=list(self.edge_map.values()))
        else:
            self.weights = dict(weight=None)

        self.coocurence_graph = ig.Graph(
            edges=[x for x in self.edge_map],
            vertex_attrs=dict(name=list(range(1, self.label_count + 1))),
            edge_attrs=self.weights
        )

        self.partition = IGraphLabelCooccurenceClusterer.METHODS[
            self.method](self.coocurence_graph, self.weights['weight'])
        return np.array(self.partition)
