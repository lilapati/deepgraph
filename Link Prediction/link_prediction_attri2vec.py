import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification

from tensorflow import keras

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import random
random.seed(0)

data_dir = "drive/MyDrive/data/DBLP"

edgelist = pd.read_csv(
    os.path.join(data_dir, "edgeList.txt"),
    sep="\t",
    header=None,
    names=["source", "target"],
)
edgelist["label"] = "cites"  # set the edge type

feature_names = ["w_{}".format(ii) for ii in range(2476)]
node_column_names = feature_names + ["subject", "year"]
node_data = pd.read_csv(
    os.path.join(data_dir, "content.txt"), sep="\t", header=None, names=node_column_names
)

G_all_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")

nx.set_node_attributes(G_all_nx, "paper", "label")

all_node_features = node_data[feature_names]

G_all = sg.StellarGraph.from_networkx(G_all_nx, node_features=all_node_features)

print(G_all.info())

year_thresh = 2006  # the threshold year for in-sample and out-of-sample set split, which can be 2007, 2008 and 2009
subgraph_edgelist = []
for ii in range(len(edgelist)):
    source_index = edgelist["source"][ii]
    target_index = edgelist["target"][ii]
    source_year = int(node_data["year"][source_index])
    target_year = int(node_data["year"][target_index])
    if source_year < year_thresh and target_year < year_thresh:
        subgraph_edgelist.append([source_index, target_index])
subgraph_edgelist = pd.DataFrame(
    np.array(subgraph_edgelist), columns=["source", "target"]
)
subgraph_edgelist["label"] = "cites"  # set the edge type

G_sub_nx = nx.from_pandas_edgelist(subgraph_edgelist, edge_attr="label")

nx.set_node_attributes(G_sub_nx, "paper", "label")

subgraph_node_ids = sorted(list(G_sub_nx.nodes))

subgraph_node_features = node_data[feature_names].reindex(subgraph_node_ids)

G_sub = sg.StellarGraph.from_networkx(G_sub_nx, node_features=subgraph_node_features)

print(G_sub.info())

nodes = list(G_sub.nodes())
number_of_walks = 2
length = 5

unsupervised_samples = UnsupervisedSampler(
    G_sub, nodes=nodes, length=length, number_of_walks=number_of_walks
)

batch_size = 50
epochs = 6

generator = Attri2VecLinkGenerator(G_sub, batch_size)

layer_sizes = [128]
attri2vec = Attri2Vec(
    layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None
)

# Build the model and expose input and output sockets of attri2vec, for node pair inputs:
x_inp, x_out = attri2vec.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-2),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=2,
    use_multiprocessing=False,
    workers=1,
    shuffle=True,
)

x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

node_ids = node_data.index
node_gen = Attri2VecNodeGenerator(G_all, batch_size).flow(node_ids)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

year_thresh = 2006
in_sample_edges = []
out_of_sample_edges = []
for ii in range(len(edgelist)):
    source_index = edgelist["source"][ii]
    target_index = edgelist["target"][ii]
    if source_index > target_index:  # neglect edge direction for the undirected graph
        continue
    source_year = int(node_data["year"][source_index])
    target_year = int(node_data["year"][target_index])
    if source_year < year_thresh and target_year < year_thresh:
        in_sample_edges.append([source_index, target_index, 1])  # get the positive edge
        negative_target_index = random.choices(
                                  node_data.index.tolist(), k=1
                                )  # generate negative node
        in_sample_edges.append(
            [source_index, negative_target_index[0], 0]
        )  # get the negative edge
    else:
        out_of_sample_edges.append(
            [source_index, target_index, 1]
        )  # get the positive edge
        negative_target_index = random.choices(
            node_data.index.tolist(), k=1
        )  # generate negative node
        out_of_sample_edges.append(
            [source_index, negative_target_index[0], 0]
        )  # get the negative edge
in_sample_edges = np.array(in_sample_edges)
out_of_sample_edges = np.array(out_of_sample_edges)

in_sample_edge_feat_from_emb = (
    node_embeddings[in_sample_edges[:, 0]] - node_embeddings[in_sample_edges[:, 1]]
) ** 2
out_of_sample_edge_feat_from_emb = (
    node_embeddings[out_of_sample_edges[:, 0]]
    - node_embeddings[out_of_sample_edges[:, 1]]
) ** 2

clf_edge_pred_from_emb = LogisticRegression(
    verbose=0, solver="lbfgs", multi_class="auto", max_iter=500
)
clf_edge_pred_from_emb.fit(in_sample_edge_feat_from_emb, in_sample_edges[:, 2])

edge_pred_from_emb = clf_edge_pred_from_emb.predict_proba(
    out_of_sample_edge_feat_from_emb
)

if clf_edge_pred_from_emb.classes_[0] == 1:
    positive_class_index = 0
else:
    positive_class_index = 1

roc_auc_score(out_of_sample_edges[:, 2], edge_pred_from_emb[:, positive_class_index])

in_sample_edge_rep_from_feat = (
    node_data[feature_names].values[in_sample_edges[:, 0]]
    - node_data[feature_names].values[in_sample_edges[:, 1]]
) ** 2
out_of_sample_edge_rep_from_feat = (
    node_data[feature_names].values[out_of_sample_edges[:, 0]]
    - node_data[feature_names].values[out_of_sample_edges[:, 1]]
) ** 2

clf_edge_pred_from_feat = LogisticRegression(
    verbose=0, solver="lbfgs", multi_class="auto", max_iter=500
)
clf_edge_pred_from_feat.fit(in_sample_edge_rep_from_feat, in_sample_edges[:, 2])

edge_pred_from_feat = clf_edge_pred_from_feat.predict_proba(
    out_of_sample_edge_rep_from_feat
)

if clf_edge_pred_from_feat.classes_[0] == 1:
    positive_class_index = 0
else:
    positive_class_index = 1

roc_auc_score(out_of_sample_edges[:, 2], edge_pred_from_feat[:, positive_class_index])