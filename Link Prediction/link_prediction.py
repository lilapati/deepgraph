# from rdflib import Graph, URIRef, Literal
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
from stellargraph import datasets
from stellargraph.data import EdgeSplitter
from stellargraph.layer import GCN, LinkEmbedding, GraphSAGE, link_classification, Attri2Vec, HinSAGE, \
    link_regression
from stellargraph.mapper import FullBatchLinkGenerator, GraphSAGELinkGenerator, \
    Attri2VecLinkGenerator, HinSAGELinkGenerator
from tensorflow.keras import layers, optimizers, regularizers, Model, losses


def get_data(dataset_name):
    if dataset_name == 'cora':
        return datasets.Cora().load()
    elif dataset_name == 'movielens':
        return datasets.MovieLens().load()
    elif dataset_name == 'citeseer':
        return datasets.CiteSeer().load()
    elif dataset_name == 'pubmed':
        return datasets.PubMedDiabetes().load()


def get_gcn_model(generator):
    gcn_model = GCN(
        layer_sizes=[16, 16],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.3,
    )
    x_in, x_out = gcn_model.in_out_tensors()
    predictions = LinkEmbedding(activation='relu')(x_out)
    predictions = layers.Reshape((-1,))(predictions)

    model = Model(inputs=x_in, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.01), loss='binary_crossentropy', metrics=["acc"])

    return model


def get_graphsage_model(generator):
    layer_sizes = [20, 20]
    graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.3)

    x_inp, x_out = graphsage.in_out_tensors()

    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)

    model = Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    return model


def get_attri2vec_model(generator):
    layer_sizes = [128]
    attri2vec = Attri2Vec(
        layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None
    )
    x_inp, x_out = attri2vec.in_out_tensors()

    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    model = Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    return model


def get_hinsage_model(generator):
    layer_sizes = [32, 32]
    hinsage = HinSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0
    )
    x_inp, x_out = hinsage.in_out_tensors()

    prediction = link_regression(
        edge_embedding_method="concat"
    )(x_out)

    model = Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    return model

def main(model_name, dataset):

    # Loading the network data
    graphs, _ = get_data(dataset)

    # Define an edge splitter on the original graph G:
    edge_splitter_test = EdgeSplitter(graphs)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    edge_splitter_train = EdgeSplitter(G_test)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
    # reduced graph G_train with the sampled links removed:
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )


    if model_name == 'gcn':
        train_gen = FullBatchLinkGenerator(G_train, method="gcn")
        train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

        test_gen = FullBatchLinkGenerator(G_test, method="gcn")
        test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
        model = get_gcn_model(train_gen)
    elif model_name == 'graphsage':
        batch_size = 32
        num_samples = [20, 10]
        train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
        train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

        test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
        test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
        model = get_graphsage_model(train_gen)
    elif model_name == 'hinsage':
        batch_size = 32
        num_samples = [8, 4]
        train_gen = HinSAGELinkGenerator(G_train, batch_size, num_samples)
        train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

        test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
        test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
        model = get_hinsage_model(train_gen)
    elif model_name == 'attri2vec':
        batch_size = 32
        train_gen = Attri2VecLinkGenerator(G_train, batch_size)
        train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

        test_gen = Attri2VecLinkGenerator(G_test, batch_size)
        test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
        model = get_attri2vec_model(train_gen)

    epochs = 100

    history = model.fit(
        train_flow, epochs=epochs, verbose=1, validation_data=test_flow, shuffle=False,
    )

    test_metrics = model.evaluate(test_flow)

    history.history['test_loss'] = test_metrics[0]
    history.history['test_acc'] = test_metrics[1]

    with open(model_name + '_' + dataset + '_history.json', 'wt') as f:
        json.dump(history.history, f, indent=4)


if __name__ == '__main__':
    for model, dataset in [(model, dataset) for model in ['hinsage'] for dataset in ['cora', 'pubmed', 'citeseer']]:
        print(model, dataset, ':')
        main(model, dataset)
