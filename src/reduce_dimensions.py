import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import *


def reduce_dimension(dataset, algorithm: str, nb_of_dimention: int, out_file=None, plot_curve=False, labels=None):

    if type(dataset) == str:
        df = pd.read_csv(dataset, ',', header=None)
    elif type(dataset) == pd.DataFrame:
        df = dataset
    else:
        try:
            df = pd.DataFrame(dataset)
        except ValueError:
            raise TypeError("dataset must be a pandas dataframe or a path to a csv file")

    if algorithm == "PCA":
        algo = PCA(n_components=nb_of_dimention,
                   svd_solver='full')
    elif algorithm == 'ICA':
        algo = FastICA(n_components=nb_of_dimention)
    elif algorithm == 'random':
        algo = GaussianRandomProjection(n_components=nb_of_dimention)
    elif algorithm == 'LDA':
        algo = LinearDiscriminantAnalysis(n_components=nb_of_dimention,
                                          solver='eigen',
                                          shrinkage='auto')
        if labels is None:
            raise ValueError("Labels must be given when using LDA algorithm (but are use less in other case)")
    else:
        raise ValueError("Unknow algorithm, please use one of: 'PCA', 'ICA', 'LDA' or 'random'")

    algo.fit(X=df,
             y=labels)

    tdf = algo.transform(X=df)
    tdf = pd.DataFrame(tdf, columns=list(map(str, range(nb_of_dimention))))

    if out_file is not None:
        tdf.to_csv(out_file, sep=',', encoding='utf-8')

    if plot_curve:
        if algorithm == 'PCA':
            eigenvalues = algo.explained_variance_
            score_list = np.zeros((2, len(eigenvalues)), dtype=np.float64)
            old_sum = 0.0
            for i, v in enumerate(eigenvalues):
                old_sum += v
                score_list[0, i] = old_sum
                score_list[1, i] = v
            plt.figure(1)
            plt.plot(range(len(eigenvalues)), score_list[0] / np.max(score_list[0]))
            plt.plot(range(len(eigenvalues)), score_list[1] / np.max(score_list[1]))
            plt.show()
        else:
            raise ValueError('Curve not yet implemented for other algorithm than PCA')
    return tdf, algo


def get_data(number_of_feature: int, algorithm: str):
    TRAIN_DATASET_LENGHT = 1951
    sc = load_starcraft_to_df()
    label = sc.get('label')
    sc = sc.drop('label', 1)
    data, algo = reduce_dimension(sc[:TRAIN_DATASET_LENGHT],
                                  algorithm,
                                  number_of_feature,
                                  labels=label[:TRAIN_DATASET_LENGHT]
                                  )
    x_train = data[:TRAIN_DATASET_LENGHT]
    y_train = label[:TRAIN_DATASET_LENGHT]

    x_test = algo.transform(X=sc[TRAIN_DATASET_LENGHT:])
    y_test = label[TRAIN_DATASET_LENGHT:]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # iris = load_iris_to_df()
    # iris = iris.drop('label', 1)
    # pca_iris = reduce_dimension(iris, 'PCA', 4, plot_curve=True)
    # ica_iris = reduce_dimension(iris, 'ICA', 2)
    sc = load_starcraft_to_df()
    labels = sc.get('label')
    sc = sc.drop('label', 1)
    print(labels.shape, sc.shape)
    pca_sc = reduce_dimension(sc, 'LDA', 70, plot_curve=False, labels=labels)

