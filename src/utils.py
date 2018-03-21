import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_iris_to_df():
    iris = load_iris()
    df = pd.DataFrame(iris['data'], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df = df.assign(label=iris['target'])
    return df


def load_starcraft_to_df():
    sc = pd.read_csv("data/starcraft_x_all.csv", header=0)
    sc_lb = pd.read_csv("data/starcraft_nl_y_all.csv", header=0)
    sc = sc.assign(label=sc_lb['label'])
    return sc


def plot_2d(df: pd.DataFrame, x_axis: str, y_axis: str, label_axis: str='label', comment=""):
    fig, ax = plt.subplots()
    ax.scatter(df.get(x_axis), df.get(y_axis), c=df.get(label_axis), cmap='gist_ncar', linewidths=1, edgecolors='black')
    ax.set_xlabel(x_axis.replace('_', ' '), fontsize=15)
    ax.set_ylabel(y_axis.replace('_', ' '), fontsize=15)
    if comment != "":
        comment += "_"
    plt.savefig("graphs/" + comment + x_axis + "_" + y_axis + "_" + label_axis + ".png")
    plt.show()



