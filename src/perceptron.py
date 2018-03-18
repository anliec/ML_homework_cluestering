import numpy as np
import pandas as pd
import os
import pickle
import sys

from keras.layers import Dense, Activation, BatchNormalization, InputLayer, Dropout
from keras.models import Sequential

from src.utils import *
from src.reduce_dimensions import get_data


def get_model(feature_count, class_count, hidden_layer, use_batch_norm=True, optimizer='rmsprop', activation='sigmoid'):
    return_model = Sequential()
    return_model.add(InputLayer(input_shape=(feature_count,)))
    if use_batch_norm:
        return_model.add(BatchNormalization())
    for neurones in hidden_layer:
        return_model.add(Dense(neurones))
        return_model.add(Activation(activation))
        if use_batch_norm:
            return_model.add(BatchNormalization())
    return_model.add(Dense(class_count))
    return_model.add(Activation('softmax'))

    return_model.compile(optimizer=optimizer,
                         loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])

    # return_model.summary()

    return return_model


def perceptron(number_of_feature, dim_reduction_algrithm, number_of_epoch, hidden_neurons=((35,),),
               use_batch_norm_values=(True,), optimizer_values=('rmsprop',),
               activation_values=('sigmoid',), training_sizes=(-1,), out_file_name=None):

    df = pd.DataFrame()
    for num_feature in number_of_feature:
        for algo in dim_reduction_algrithm:
            x_learn, y_learn, x_test, y_test = get_data(num_feature, algo)
            for opt in optimizer_values:
                for act in activation_values:
                    for use_batch_norm in use_batch_norm_values:
                        for layers in hidden_neurons:
                            for train_limit in training_sizes:
                                number_of_feature = len(x_learn[0])
                                number_of_class = len(y_learn[0])
                                model = get_model(number_of_feature, number_of_class, layers, use_batch_norm, opt, act)
                                h = model.fit(x=np.array(x_learn[:train_limit]),
                                              y=np.array(y_learn[:train_limit]),
                                              batch_size=len(x_learn[:train_limit]),
                                              epochs=number_of_epoch,
                                              validation_data=(np.array(x_test),
                                                               np.array(y_test)),
                                              verbose=0
                                              )
                                epoch = h.epoch
                                h_values = h.history.values()
                                values = np.array([epoch, ] + list(h_values))
                                tmp = pd.DataFrame(data=values.T, columns=["epoch", ] + list(h.history.keys()))
                                tmp.assign(use_batch_norm=pd.Series([use_batch_norm] * number_of_epoch),
                                           optimizer=pd.Series([opt] * number_of_epoch),
                                           activation=pd.Series([act] * number_of_epoch),
                                           layers=pd.Series([str(layers)] * number_of_epoch),
                                           train_size=pd.Series([str(train_limit)] * number_of_epoch),
                                           reduction_method=pd.Series([algo] * number_of_epoch),
                                           number_of_feature=pd.Series([num_feature] * number_of_epoch))
                                if out_file_name is None:
                                    df = df.append(tmp)
                                else:
                                    path = "stats/" + out_file_name + ".csv"
                                    if not os.path.exists("stats"):
                                        os.makedirs("stats")
                                    if os.path.exists("stats/" + data_set_name + ".csv"):
                                        tmp.to_csv(path_or_buf=path, mode='a', header=False)
                                    else:
                                        tmp.to_csv(path_or_buf=path)
    return df


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Wrong command line argument !")
    #     print()
    #     print("Expected call is:", sys.argv[0], "[dataset]")
    #     print("dataset: one of \"creditcard\" or \"starcraft\"")
    #     exit(1)
    # data_set_name = sys.argv[1]
    # if data_set_name == "starcraft":
    result = perceptron(number_of_feature=range(5, 31, 5),
                        dim_reduction_algrithm=('PCA', 'ICA', 'random'),
                        number_of_epoch=600,
                        hidden_neurons=((), (35,), (25,), (15,), (10,), (30, 10), (20, 10)),
                        use_batch_norm_values=(True,),
                        optimizer_values=('rmsprop',),
                        activation_values=('sigmoid',),  # , 'relu', 'linear', 'selu'),
                        training_sizes=(-1,),
                        out_file_name="perceptron"
                        )
    # else:
    #     raise ValueError("unknow dataset:", data_set_name)
    # if not os.path.exists("stats"):
    #     os.makedirs("stats")
    # result.to_csv(path_or_buf="stats/per_" + data_set_name + ".csv")

