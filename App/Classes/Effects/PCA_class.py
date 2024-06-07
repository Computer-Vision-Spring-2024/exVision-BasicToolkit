import numpy as np


class PCA_class:
    """
    Principal Component Analysis (PCA) class.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep.
    svd_solver : str, optional
        Solver to use for the decomposition. Currently not used.
    """

    def __init__(self, n_components=None, svd_solver="full"):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.mean = None
        self.components = None
        self.explained_variance_ratio_ = None

    def fit(self, X, method="svd"):
        """
        Fit the model with X using the specified method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        method : str, optional
            Method to use for the decomposition ('svd' or 'eigen').
        """
        # Mean centering
        self.mean = np.mean(X, axis=0)  # Compute the mean of X
        X = X - self.mean  # Subtract the mean from X

        # Handle number of components
        if self.n_components is None:
            self.n_components = min(X.shape) - 1

        if method == "svd":
            # Compute SVD
            U, S, Vt = np.linalg.svd(X, full_matrices=False)  # Perform SVD on X

            # Compute explained variance ratio
            explained_variance_ = (S**2) / (
                X.shape[0] - 1
            )  # Compute the explained variance
            total_variance = explained_variance_.sum()  # Compute the total variance
            explained_variance_ratio_ = (
                explained_variance_ / total_variance
            )  # Compute the explained variance ratio

            self.components = Vt[
                : self.n_components
            ]  # Keep the first n_components components
            self.explained_variance_ratio_ = explained_variance_ratio_[
                : self.n_components
            ]  # Keep the explained variance ratio for the first n_components

        elif method == "eigen":
            # Compute covariance matrix
            covariance_matrix = np.dot(X.T, X)  # Compute the covariance matrix of X

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(
                covariance_matrix
            )  # Compute the eigenvalues and eigenvectors of the covariance matrix

            # Sort eigenvalues and eigenvectors by decreasing eigenvalues
            idx = eigenvalues.argsort()[
                ::-1
            ]  # Get the indices that would sort the eigenvalues in decreasing order
            eigenvalues = eigenvalues[idx]  # Sort the eigenvalues
            eigenvectors = eigenvectors[:, idx]  # Sort the eigenvectors accordingly

            # Compute explained variance ratio
            total_variance = eigenvalues.sum()  # Compute the total variance
            explained_variance_ratio_ = (
                eigenvalues / total_variance
            )  # Compute the explained variance ratio

            self.components = eigenvectors[
                :, : self.n_components
            ].T  # Keep the first n_components components
            self.explained_variance_ratio_ = explained_variance_ratio_[
                : self.n_components
            ]  # Keep the explained variance ratio for the first n_components

        else:
            raise ValueError("Invalid method. Expected 'svd' or 'eigen'.")
        return self

    def project(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        X = X - self.mean  # Mean centering
        return np.dot(X, self.components.T)  # Project X onto the principal components

    def fit_transform(self, X, method="svd"):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        method : str, optional
            Method to use for the decomposition ('svd' or 'eigen').

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X, method)  # Fit the model with X
        return self.transform(X)  # Apply the dimensionality reduction on X
