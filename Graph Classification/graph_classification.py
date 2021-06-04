import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification

from stellargraph import datasets

from sklearn import model_selection

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import json


def get_data(dataset_name='MUTAG'):
    if dataset_name == 'MUTAG':
        return datasets.MUTAG().load()
    else:
        return datasets.PROTEINS().load()


def get_gcn_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model


def get_dgcnn_model(generator):
    layer_sizes = [32, 32, 32, 1]
    dgcnn_model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh"],
        k=35,
        bias=False,
        generator=generator,
    )
    x_inp, x_out = dgcnn_model.in_out_tensors()

    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)

    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

    x_out = Flatten()(x_out)

    x_out = Dense(units=128, activation="relu")(x_out)
    x_out = Dropout(rate=0.5)(x_out)

    predictions = Dense(units=1, activation="sigmoid")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(
        optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"],
    )

    return model


def main(model_name='gcn', dataset='MUTAG'):
    graphs, graph_labels = get_data(dataset)

    graph_labels.value_counts().to_frame()

    graph_labels = pd.get_dummies(graph_labels, drop_first=True)
    generator = PaddedGraphGenerator(graphs=graphs)

    train_graphs, test_graphs = model_selection.train_test_split(
        graph_labels, train_size=0.9, test_size=None, stratify=graph_labels,
    )

    train_gen = generator.flow(
        list(train_graphs.index - 1),
        targets=train_graphs.values,
        batch_size=50,
        symmetric_normalization=False,
    )

    test_gen = generator.flow(
        list(test_graphs.index - 1),
        targets=test_graphs.values,
        batch_size=1,
        symmetric_normalization=False,
    )

    epochs = 100

    if model_name == 'gcn':
        model = get_gcn_model(generator)
    else:
        model = get_dgcnn_model(generator)

    history = model.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
    )

    test_metrics = model.evaluate(test_gen)

    history.history['test_loss'] = test_metrics[0]
    history.history['test_acc'] = test_metrics[1]

    with open(model_name + '_' + dataset + '_history.json', 'wt') as f:
        json.dump(history.history, f, indent=4)


if __name__ == '__main__':
    # main('gcn','MUTAG')
    # main('gcn','PROTEINS')
    main('dgcnn','MUTAG')
    main('dgcnn','PROTEINS')