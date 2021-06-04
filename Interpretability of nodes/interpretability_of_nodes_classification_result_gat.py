import networkx as nx
import pandas as pd
import numpy as np
from scipy import stats
import os
import time
import sys
import stellargraph as sg
from copy import deepcopy


from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT, GraphAttention

from tensorflow.keras import layers, optimizers, losses, metrics, models, Model
from sklearn import preprocessing, feature_extraction, model_selection
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from stellargraph import datasets
from IPython.display import display, HTML

dataset = datasets.Cora()
display(HTML(dataset.description))
G, subjects = dataset.load()

print(G.info())

train_subjects, test_subjects = model_selection.train_test_split(
    subjects, train_size=140, test_size=None, stratify=subjects
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=500, test_size=None, stratify=test_subjects
)

from collections import Counter

Counter(train_subjects)

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)

all_targets = target_encoding.transform(subjects)

generator = FullBatchNodeGenerator(G, method="gat", sparse=False)

train_gen = generator.flow(train_subjects.index, train_targets)

gat = GAT(
    layer_sizes=[8, train_targets.shape[1]],
    attn_heads=8,
    generator=generator,
    bias=True,
    in_dropout=0,
    attn_dropout=0,
    activations=["elu", "softmax"],
    normalize=None,
    saliency_map_support=True,
)

x_inp, predictions = gat.in_out_tensors()

model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.categorical_crossentropy,
    weighted_metrics=["acc"],
)

val_gen = generator.flow(val_subjects.index, val_targets)

N = G.number_of_nodes()
history = model.fit(
    train_gen, validation_data=val_gen, shuffle=False, epochs=10, verbose=2
)

sg.utils.plot_history(history)

test_gen = generator.flow(test_subjects.index, test_targets)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

model_json = model.to_json()
model_weights = model.get_weights()

model2 = models.model_from_json(model_json, custom_objects=sg.custom_keras_layers)
model2.set_weights(model_weights)
model2_weights = model2.get_weights()

pred2 = model2.predict(test_gen)
pred1 = model.predict(test_gen)
print(np.allclose(pred1, pred2))

from stellargraph.interpretability.saliency_maps import IntegratedGradientsGAT
from stellargraph.interpretability.saliency_maps import GradientSaliencyGAT

graph_nodes = list(G.nodes())
all_gen = generator.flow(graph_nodes)
target_nid = 1109199
target_idx = graph_nodes.index(target_nid)
target_gen = generator.flow([target_nid])

y_true = all_targets[target_idx]  # true class of the target node

y_pred = model.predict(target_gen).squeeze()
class_of_interest = np.argmax(y_pred)
print(
    "target node id: {}, \ntrue label: {}, \npredicted label: {}".format(
        target_nid, y_true, y_pred.round(2)
    )
)

int_grad_saliency = IntegratedGradientsGAT(model, train_gen, generator.node_list)
saliency = GradientSaliencyGAT(model, train_gen)

G_ego = nx.ego_graph(G.to_networkx(), target_nid, radius=len(gat.activations))

integrate_link_importance = int_grad_saliency.get_link_importance(
    target_nid, class_of_interest, steps=25
)
print("integrated_link_mask.shape = {}".format(integrate_link_importance.shape))

integrated_node_importance = int_grad_saliency.get_node_importance(
    target_nid, class_of_interest, steps=25
)
print("\nintegrated_node_importance", integrated_node_importance.round(2))
print(
    "integrated self-importance of target node {}: {}".format(
        target_nid, integrated_node_importance[target_idx].round(2)
    )
)
print(
    "\nEgo net of target node {} has {} nodes".format(target_nid, G_ego.number_of_nodes())
)
print(
    "Number of non-zero elements in integrated_node_importance: {}".format(
        np.count_nonzero(integrated_node_importance)
    )
)

sorted_indices = np.argsort(integrate_link_importance.flatten().reshape(-1))
sorted_indices = np.array(sorted_indices)
integrated_link_importance_rank = [(int(k / N), k % N) for k in sorted_indices[::-1]]

topk = 10
print(
    "Top {} most important links by integrated gradients are {}".format(
        topk, integrated_link_importance_rank[:topk]
    )
)

nx.set_node_attributes(G_ego, values={x[0]: {"subject": x[1]} for x in subjects.items()})

node_size_factor = 1e2
link_width_factor = 4

nodes = list(G_ego.nodes())
colors = pd.DataFrame(
    [v[1]["subject"] for v in G_ego.nodes(data=True)], index=nodes, columns=["subject"]
)
colors = np.argmax(target_encoding.transform(colors), axis=1) + 1

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
pos = nx.spring_layout(G_ego)
# Draw ego as large and red
node_sizes = [integrated_node_importance[graph_nodes.index(k)] for k in nodes]
node_shapes = [
    "o" if integrated_node_importance[graph_nodes.index(k)] > 0 else "d" for k in nodes
]
positive_colors, negative_colors = [], []
positive_node_sizes, negative_node_sizes = [], []
positive_nodes, negative_nodes = [], []
# node_size_sclae is used for better visualization of nodes
node_size_scale = node_size_factor / np.max(node_sizes)
for k in range(len(node_shapes)):
    if list(nodes)[k] == target_nid:
        continue
    if node_shapes[k] == "o":
        positive_colors.append(colors[k])
        positive_nodes.append(list(nodes)[k])
        positive_node_sizes.append(node_size_scale * node_sizes[k])

    else:
        negative_colors.append(colors[k])
        negative_nodes.append(list(nodes)[k])
        negative_node_sizes.append(node_size_scale * abs(node_sizes[k]))

cmap = plt.get_cmap("jet", np.max(colors) - np.min(colors) + 1)
nc = nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=positive_nodes,
    node_color=positive_colors,
    cmap=cmap,
    node_size=positive_node_sizes,
    with_labels=False,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    node_shape="o",
)
nc = nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=negative_nodes,
    node_color=negative_colors,
    cmap=cmap,
    node_size=negative_node_sizes,
    with_labels=False,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    node_shape="d",
)
# Draw the target node as a large star colored by its true subject
nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=[target_nid],
    node_size=50 * abs(node_sizes[nodes.index(target_nid)]),
    node_shape="*",
    node_color=[colors[nodes.index(target_nid)]],
    cmap=cmap,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    label="Target",
)

edges = G_ego.edges()
# link_width_scale is used for better visualization of links
weights = [
    integrate_link_importance[graph_nodes.index(u), graph_nodes.index(v)]
    for u, v in edges
]
link_width_scale = link_width_factor / np.max(weights)
edge_colors = [
    "red"
    if integrate_link_importance[graph_nodes.index(u), graph_nodes.index(v)] > 0
    else "blue"
    for u, v in edges
]

ec = nx.draw_networkx_edges(
    G_ego, pos, edge_color=edge_colors, width=[link_width_scale * w for w in weights]
)
plt.legend()
plt.colorbar(nc, ticks=np.arange(np.min(colors), np.max(colors) + 1))
plt.axis("off")
plt.show()

[X, _, A], y_true_all = all_gen[0]
N = A.shape[-1]
X_bk = deepcopy(X)
edges = [(graph_nodes.index(u), graph_nodes.index(v)) for u, v in G_ego.edges()]
nodes_idx = [graph_nodes.index(v) for v in nodes]
selected_nodes = np.array([[target_idx]], dtype="int32")
clean_prediction = model.predict([X, selected_nodes, A]).squeeze()
predict_label = np.argmax(clean_prediction)
groud_truth_edge_importance = np.zeros((N, N), dtype="float")
groud_truth_node_importance = []

for node in nodes_idx:
    if node == target_idx:
        groud_truth_node_importance.append(0)
        continue
    X = deepcopy(X_bk)
    # we set all the features of the node to zero to check the ground truth node importance.
    X[0, node, :] = 0
    predict_after_perturb = model.predict([X, selected_nodes, A]).squeeze()
    prediction_change = (
        clean_prediction[predict_label] - predict_after_perturb[predict_label]
    )
    groud_truth_node_importance.append(prediction_change)

node_shapes = [
    "o" if groud_truth_node_importance[k] > 0 else "d" for k in range(len(nodes))
]
positive_colors, negative_colors = [], []
positive_node_sizes, negative_node_sizes = [], []
positive_nodes, negative_nodes = [], []
# node_size_scale is used for better visulization of nodes
node_size_scale = node_size_factor / max(groud_truth_node_importance)

for k in range(len(node_shapes)):
    if nodes_idx[k] == target_idx:
        continue
    if node_shapes[k] == "o":
        positive_colors.append(colors[k])
        positive_nodes.append(graph_nodes[nodes_idx[k]])
        positive_node_sizes.append(node_size_scale * groud_truth_node_importance[k])
    else:
        negative_colors.append(colors[k])
        negative_nodes.append(graph_nodes[nodes_idx[k]])
        negative_node_sizes.append(node_size_scale * abs(groud_truth_node_importance[k]))

X = deepcopy(X_bk)
for edge in edges:
    original_val = A[0, edge[0], edge[1]]
    if original_val == 0:
        continue
    # we set the weight of a given edge to zero to check the ground truth link importance
    A[0, edge[0], edge[1]] = 0
    predict_after_perturb = model.predict([X, selected_nodes, A]).squeeze()
    groud_truth_edge_importance[edge[0], edge[1]] = (
        predict_after_perturb[predict_label] - clean_prediction[predict_label]
    ) / (0 - 1)
    A[0, edge[0], edge[1]] = original_val
#     print(groud_truth_edge_importance[edge[0], edge[1]])

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
cmap = plt.get_cmap("jet", np.max(colors) - np.min(colors) + 1)
# Draw the target node as a large star colored by its true subject
nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=[target_nid],
    node_size=50 * abs(node_sizes[nodes_idx.index(target_idx)]),
    node_color=[colors[nodes_idx.index(target_idx)]],
    cmap=cmap,
    node_shape="*",
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    label="Target",
)
# Draw the ego net
nc = nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=positive_nodes,
    node_color=positive_colors,
    cmap=cmap,
    node_size=positive_node_sizes,
    with_labels=False,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    node_shape="o",
)

nc = nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=negative_nodes,
    node_color=negative_colors,
    cmap=cmap,
    node_size=negative_node_sizes,
    with_labels=False,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    node_shape="d",
)
edges = G_ego.edges()
# link_width_scale is used for better visulization of links
link_width_scale = link_width_factor / np.max(groud_truth_edge_importance)
weights = [
    link_width_scale
    * groud_truth_edge_importance[graph_nodes.index(u), graph_nodes.index(v)]
    for u, v in edges
]

edge_colors = [
    "red"
    if groud_truth_edge_importance[graph_nodes.index(u), graph_nodes.index(v)] > 0
    else "blue"
    for u, v in edges
]

ec = nx.draw_networkx_edges(G_ego, pos, edge_color=edge_colors, width=weights)
plt.legend()
plt.colorbar(nc, ticks=np.arange(np.min(colors), np.max(colors) + 1))
plt.axis("off")
plt.show()

