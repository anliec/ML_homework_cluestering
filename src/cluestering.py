import sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris


def cluestering(dataset, algorithm: str, n_clusters: int, out_file_name: str =None):

    if type(dataset) == str:
        df = pd.read_csv(dataset, ',', header=None)
    elif type(dataset) == pd.DataFrame:
        df = dataset
    else:
        try:
            df = pd.DataFrame(dataset)
        except ValueError:
            raise TypeError("dataset must be a pandas dataframe or a path to a csv file")

    if algorithm == "kmeans":
        algo = KMeans(n_clusters=n_clusters,
                      n_init=30,
                      n_jobs=-1
                      )
    elif algorithm == "EM":
        algo = GaussianMixture(n_components=n_clusters,
                               n_init=30,
                               init_params='kmeans'  # can also be kmeans or random !
                               )
    else:
        raise ValueError("algorithm must be one of 'kmeans' or 'EM'")

    algo.fit(X=df)

    p = algo.predict(X=df)

    dfp = df.assign(label=p)

    if out_file_name is not None:
        dfp.to_csv(out_file_name, sep=',', encoding='utf-8', header=None)

    return dfp


if __name__ == '__main__':
    iris = load_iris()
    cluestering(dataset=iris['data'], algorithm='kmeans', n_clusters=3, out_file_name="results/kmeans_iris.csv")





