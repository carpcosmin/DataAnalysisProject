'''
A class for implementing PCA (Principal Components Analysis)
'''

import numpy as np

class PCA:
    def __init__(self, X):
        self.X = X
        # compute the
        self.Cov = np.cov(m=X, rowvar=False)  # we have the variables on the columns

        # extract the eigenvalues and the eigenvectors
        # for the variance-covariance matrix
        values, vectors = np.linalg.eigh(a=self.Cov)
        print(values); print(vectors.shape)
        k_desc = [k for k in reversed(np.argsort(values))]
        print(k_desc)
        self.alpha = values[k_desc]
        print(self.alpha)
        self.a = vectors[:, k_desc]

        # compute the principal components
        self.C = self.X @ self.a
        # self.C = np.matmul(self.X, self.a)

        # compute the correlation factors (factor loadings)
        self.Rxc = self.a * np.sqrt(self.alpha)


    def getCov(self):
        return self.Cov

    def getAlpha(self):
        return self.alpha

    def getFactorLoadings(self):
        return self.Rxc

    def getComponents(self):
        return self.C

    def getScores(self):
        # compute the scores (standardized principal components)
        scores = self.C / np.sqrt(self.alpha)
        return scores
