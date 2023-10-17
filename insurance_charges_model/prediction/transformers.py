import numpy as np
from numpy import bool_, inf, int, float
from sklearn.base import BaseEstimator, TransformerMixin
import featuretools as ft


class BooleanTransformer(BaseEstimator, TransformerMixin):
    """Convert values to True or False."""

    def __init__(self, true_value="yes", false_value="no"):
        """Initialize BooleanTransformer instance."""
        self.true_value = true_value
        self.false_value = false_value

    def fit(self, X, y=None):
        """Fit the transformer to a dataset."""
        return self

    def transform(self, X, y=None):
        """Transform a dataset."""
        def f(value):
            if type(value) is bool or type(value) is bool_:
                return value
            elif value == self.true_value:
                return True
            elif value == self.false_value:
                return False
            else:
                raise ValueError("Value: {} cannot be mapped to a boolean value.".format(value))

        X = np.vectorize(f)(X)

        return X


class IntToFloatTransformer(BaseEstimator, TransformerMixin):
    """Convert integer values to floating point values."""

    def fit(self, X, y=None):
        """Fit the transformer to a dataset."""
        return self

    def transform(self, X, y=None):
        """Transform a dataset."""
        def f(value):
            if type(value) is int:
                return float(value)
            else:
                return value

        X = np.vectorize(f)(X)

        return X


class DFSTransformer(BaseEstimator, TransformerMixin):
    """Generate features using deep feature synthesis."""

    def __init__(self, target_entity, trans_primitives, ignore_variables):
        """Initialize instance of DFSTransformer."""
        self.target_entity = target_entity
        self.trans_primitives = trans_primitives
        self.ignore_variables = ignore_variables
        self.feature_defs = None

    def fit(self, X, y=None):
        """Fit the transformer to a dataset."""
        entityset = ft.EntitySet(id="Transactions")
        if "index" not in X.columns:
            entityset = entityset.entity_from_dataframe(entity_id=self.target_entity,
                                                        dataframe=X,
                                                        make_index=True,
                                                        index="index")
        else:
            entityset = entityset.entity_from_dataframe(entity_id=self.target_entity,
                                                        dataframe=X)

        feature_matrix, feature_defs = ft.dfs(entityset=entityset,
                                              target_entity=self.target_entity,
                                              trans_primitives=self.trans_primitives,
                                              ignore_variables=self.ignore_variables)

        self.feature_defs = feature_defs
        return self

    def transform(self, X, y=None):
        """Transform a dataset."""
        entityset = ft.EntitySet(id="Transactions")
        if "index" not in X.columns:
            entityset = entityset.entity_from_dataframe(entity_id=self.target_entity,
                                                        dataframe=X,
                                                        make_index=True,
                                                        index="index")
        else:
            entityset = entityset.entity_from_dataframe(entity_id=self.target_entity,
                                                        dataframe=X)

        feature_matrix = ft.calculate_feature_matrix(self.feature_defs, entityset)
        return feature_matrix


class InfinityToNaNTransformer(BaseEstimator, TransformerMixin):
    """Convert inf values to NaN values."""

    def fit(self, X, y=None):
        """Fit the transformer to a dataset."""
        return self

    def transform(self, X, y=None):
        """Transform a dataset."""
        def f(value):
            if value == inf:
                return np.nan
            else:
                return value

        X = np.vectorize(f)(X)

        return X
