import pandas as pd
import functions as fct
import pca.PCA as pca
import graphics as g

table = pd.read_csv('./dataIN/dataIN.csv', index_col=0, sep=';')
# print(table)
rows = table.index.values
cols = table.columns.values
matrix = table.values

Xstd = fct.standardize(matrix)
# print(Xstd); print(type(Xstd)); print(Xstd.shape)

Xstd_df = pd.DataFrame(data=Xstd, columns=cols,
                       index=rows)
# print(Xstd_df)
Xstd_df.to_csv('./dataOUT/Xstd.csv')

# save the variance-covariance matrix into a CSV file
pca_obj = pca.PCA(Xstd)
Cov = pca_obj.getCov()
Cov_df = pd.DataFrame(data=Cov,
        columns=cols,
        index=cols)
Cov_df.to_csv('./dataOUT/Cov.csv')
g.correlogram(matrix=Cov_df, dec=1, title='Correlogram of variance-covariance matrix')
g.show()

# create the graphic of explained variance by the principal components
eigenValues = pca_obj.getEigenValues()
g.principalComponents(eigenvalues=eigenValues,
            title='Explained variance by the principal components')
g.show()

components = ['C'+str(j+1) for j in range(cols.shape[0])]

# create the correlogram of factor loadings
Rxc = pca_obj.getFactorLoadings()
Rxc_df = pd.DataFrame(data=Rxc,
    columns=components,
    index=cols)
Rxc_df.to_csv('./dataOUT/FactorLoadings.csv')
g.correlogram(matrix=Rxc_df, dec=1,
        title='Correlogram of factor loadings')
g.show()

# create the correlogram of scores (standardized principal components)
scores = pca_obj.getScores()
scores_df = pd.DataFrame(data=scores,
        columns=components,
        index=rows)
scores_df.to_csv('./dataOUT/Scores.csv')
g.correlogram(matrix=scores_df, dec=1,
        title='Correlogram of scores')
g.show()
