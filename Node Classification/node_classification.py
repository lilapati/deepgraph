import pandas as pd
from rdflib.extras.external_graph_libs import *
# from rdflib import Graph, URIRef, Literal
from stellargraph.mapper import FullBatchNodeGenerator, RelationalFullBatchNodeGenerator
from stellargraph.layer import GCN, RGCN, GAT, PPNP
from stellargraph import datasets
import json

from tensorflow.keras import layers, optimizers, regularizers, Model
from sklearn import preprocessing, model_selection


def get_data(dataset_name):
    if dataset_name == 'cora':
        return datasets.Cora().load()
    elif dataset_name == 'aifb':
        return datasets.AIFB().load()
    elif dataset_name == 'citeseer':
        return datasets.CiteSeer().load()
    elif dataset_name == 'pubmed':
        return datasets.PubMedDiabetes().load()


def get_gcn_model(generator, size):
    gcn_model = GCN(
        layer_sizes=[16, 16],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_in, x_out = gcn_model.in_out_tensors()
    predictions = layers.Dense(units=size, activation='softmax')(x_out)

    model = Model(inputs=x_in, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.005), loss='categorical_crossentropy', metrics=["acc"])

    return model


def get_rgcn_model(generator, size):
    rgcn_model = RGCN(
        layer_sizes=[32, 32],
        activations=["relu", "relu"],
        generator=generator,
        bias=True,
        num_bases=20,
        dropout=0.5,
    )

    x_in, x_out = rgcn_model.in_out_tensors()
    predictions = layers.Dense(units=size, activation='softmax')(x_out)

    model = Model(inputs=x_in, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.005), loss='categorical_crossentropy', metrics=["acc"])

    return model


def get_gat_model(generator, size):
    gat_model = GAT(
        layer_sizes=[8, size],
        activations=["elu", "softmax"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
    )
    x_in, predictions = gat_model.in_out_tensors()
    model = Model(inputs=x_in, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.005), loss='categorical_crossentropy', metrics=["acc"])

    return model


def get_sgc_model(generator, size):
    sgc_model = GCN(
        layer_sizes=[size],
        activations=['softmax'],
        generator=generator,
        bias=True,
        kernel_regularizer=regularizers.l2(5e-4),
        dropout=0.5,
    )
    x_in, predictions = sgc_model.in_out_tensors()
    model = Model(inputs=x_in, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.005), loss='categorical_crossentropy', metrics=["acc"])

    return model


def get_ppnp_model(generator, size):
    ppnp_model = PPNP(
        layer_sizes=[64, 64, size],
        activations=['relu', 'relu', 'relu'],
        generator=generator,
        kernel_regularizer=regularizers.l2(0.001),
        dropout=0.5,
    )
    x_in, x_out = ppnp_model.in_out_tensors()
    predictions = layers.Softmax()(x_out)
    model = Model(inputs=x_in, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.005), loss='categorical_crossentropy', metrics=["acc"])
    return model


def main(model_name, dataset):

    # Loading the network data
    graphs, node_subjects = get_data(dataset)
    # print(graphs.info())

    # print(node_subjects.value_counts().to_frame())

    node_subjects = pd.get_dummies(node_subjects, drop_first=True)

    # Splitting the data
    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=0.8, test_size=None
    )

    # print(train_subjects.value_counts().to_frame())

    val_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=0.5, stratify=test_subjects
    )

    # Converting to numeric arrays
    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)

    if model_name == 'gcn':
        generator = FullBatchNodeGenerator(graphs, method='gcn')
        model = get_gcn_model(generator, train_targets.shape[1])
    elif model_name == 'rgcn':
        generator = RelationalFullBatchNodeGenerator(graphs, sparse=True)
        model = get_rgcn_model(generator, train_targets.shape[1])
    elif model_name == 'gat':
        generator = FullBatchNodeGenerator(graphs, method='gat')
        model = get_gat_model(generator, train_targets.shape[1])
    elif model_name == 'sgc':
        generator = FullBatchNodeGenerator(graphs, method='sgc', k=2)
        model = get_sgc_model(generator, train_targets.shape[1])
    elif model_name == 'ppnp':
        generator = FullBatchNodeGenerator(graphs, method='ppnp', sparse=False, teleport_probability=0.1)
        model = get_ppnp_model(generator, train_targets.shape[1])

    train_gen = generator.flow(train_subjects.index, train_targets)
    val_gen = generator.flow(val_subjects.index, val_targets)
    test_gen = generator.flow(test_subjects.index, test_targets)

    epochs = 100

    history = model.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=val_gen, shuffle=False,
    )

    test_metrics = model.evaluate(test_gen)

    history.history['test_loss'] = test_metrics[0]
    history.history['test_acc'] = test_metrics[1]

    with open(model_name + '_' + dataset + '_history.json', 'wt') as f:
        json.dump(history.history, f, indent=4)


if __name__ == '__main__':
    for model, dataset in [(model, dataset) for model in ['ppnp'] for dataset in ['pubmed', 'citeseer', 'cora', 'aifb']]:
        print(model, dataset, ':')
        main(model, dataset)
