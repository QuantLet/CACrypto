import pandas as pd
from ipca import InstrumentedPCA
import numpy as np

class IPCA_custom(InstrumentedPCA):
    def updateFactorsOOS(self, X=None, y=None, indices=None):
        X, y, indices, metad = _prep_input(X, y, indices)
        N, L, T = metad["N"], metad["L"], metad["T"]
        ypred = np.full((N), np.nan)

        # Compute realized factor returns
        Numer = self.Gamma.T.dot(X.T).dot(y)
        Denom = self.Gamma.T.dot(X.T).dot(X).dot(self.Gamma)
        Factor_OOS = np.linalg.solve(Denom, Numer.reshape((-1, 1)))

        self.Factors = Factor_OOS

def _prep_input(X, y=None, indices=None):
    """handle mapping from different inputs type to consistent internal data
    Parameters
    ----------
    X :  numpy array or pandas DataFrame
        matrix of characteristics where each row corresponds to a
        entity-time pair in indices.  The number of characteristics
        (columns here) used as instruments is L.
        If given as a DataFrame, we assume that it contains a mutliindex
        mapping to each entity-time pair
    y : numpy array or pandas Series, optional
        dependent variable where indices correspond to those in X
        If given as a Series, we assume that it contains a mutliindex
        mapping to each entity-time pair
    indices : numpy array or pandas MultiIndex, optional
        array containing the panel indices.  Should consist of two
        columns:
        - Column 1: entity id (i)
        - Column 2: time index (t)
        The panel may be unbalanced. The number of unique entities is
        n_samples, the number of unique dates is T, and the number of
        characteristics used as instruments is L.
    Returns
    -------
    X :  numpy array
        matrix of characteristics where each row corresponds to a
        entity-time pair in indices.  The number of characteristics
        (columns here) used as instruments is L.
    y : numpy array
        dependent variable where indices correspond to those in X
    indices : numpy array
        array containing the panel indices.  Should consist of two
        columns:
        - Column 1: entity id (i)
        - Column 2: time index (t)
        The panel may be unbalanced. The number of unique entities is
        n_samples, the number of unique dates is T, and the number of
        characteristics used as instruments is L.
    metad : dict
        contains metadata on inputs:
        dates : array-like
            unique dates in panel
        ids : array-like
            unique ids in panel
        chars : array-like
            labels for X chars/columns
        T : scalar
            number of time periods
        N : scalar
            number of ids
        L : scalar
            total number of characteristics
    """

    # Check panel input
    if X is None:
        raise ValueError('Must pass panel input data.')
    else:
        # remove panel rows containing missing obs
        non_nan_ind = ~np.any(np.isnan(X), axis=1)
        X = X[non_nan_ind]
        if y is not None:
            y = y[non_nan_ind]

    # if data-frames passed, break out indices from data
    if isinstance(X, pd.DataFrame) and not isinstance(y, pd.Series):
        indices = X.index
        chars = X.columns
        X = X.values
    elif not isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        indices = y.index
        y = y.values
        chars = np.arange(X.shape[1])
    elif isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        Xind = X.index
        chars = X.columns
        yind = y.index
        X = X.values
        y = y.values
        if not np.array_equal(Xind, yind):
            raise ValueError("If indices are provided with both X and y\
                              they be the same")
        indices = Xind
    else:
        chars = np.arange(X.shape[1])

    if indices is None:
        raise ValueError("entity-time indices must be provided either\
                          separately or as a MultiIndex with X/y")

    # extract numpy array and labels from multiindex
    if isinstance(indices, pd.MultiIndex):
        indices = indices.to_frame().values
    ids = np.unique(indices[:, 0])
    dates = np.unique(indices[:, 1])
    indices[:,0] = np.unique(indices[:,0], return_inverse=True)[1]
    indices[:,1] = np.unique(indices[:,1], return_inverse=True)[1]

    # init data dimensions
    T = np.size(dates, axis=0)
    N = np.size(ids, axis=0)
    L = np.size(chars, axis=0)

    # prep metadata
    metad = {}
    metad["dates"] = dates
    metad["ids"] = ids
    metad["chars"] = chars
    metad["T"] = T
    metad["N"] = N
    metad["L"] = L

    return X, y, indices, metad
