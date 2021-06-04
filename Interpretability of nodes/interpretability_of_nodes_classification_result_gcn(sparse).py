import networkx as nx
import pandas as pd
import numpy as np
from scipy import stats
import os
import time
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics, Model, regularizers
from sklearn import preprocessing, feature_extraction, model_selection
from copy import deepcopy
import matplotlib.pyplot as plt
from stellargraph import datasets
from IPython.display import display, HTML

dataset = datasets.Cora()
display(HTML(dataset.description))
G, subjects = dataset.load()

train_subjects, test_subjects = model_selection.train_test_split(
    subjects, train_size=140, test_size=None, stratify=subjects
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=500, test_size=None, stratify=test_subjects
)

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)

all_targets = target_encoding.transform(subjects)

generator = FullBatchNodeGenerator(G, sparse=True)

train_gen = generator.flow(train_subjects.index, train_targets)

layer_sizes = [16, 16]
gcn = GCN(
    layer_sizes=layer_sizes,
    activations=["elu", "elu"],
    generator=generator,
    dropout=0.3,
    kernel_regularizer=regularizers.l2(5e-4),
)

x_inp, x_out = gcn.in_out_tensors()
x_out = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = keras.Model(inputs=x_inp, outputs=x_out)

model.compile(
    optimizer=optimizers.Adam(lr=0.01),  # decay=0.001),
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy],
)

val_gen = generator.flow(val_subjects.index, val_targets)

history = model.fit(
    train_gen, shuffle=False, epochs=20, verbose=2, validation_data=val_gen
)

sg.utils.plot_history(history)

test_gen = generator.flow(test_subjects.index, test_targets)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

from stellargraph.interpretability.saliency_maps import IntegratedGradients

graph_nodes = list(G.nodes())
target_nid = 1109199
target_idx = graph_nodes.index(target_nid)
y_true = all_targets[target_idx]

all_gen = generator.flow(graph_nodes)
y_pred = model.predict(all_gen)[0, target_idx]
class_of_interest = np.argmax(y_pred)

print(
    "Selected node id: {}, \nTrue label: {}, \nPredicted scores: {}".format(
        target_nid, y_true, y_pred.round(2)
    )
)

int_grad_saliency = IntegratedGradients(model, train_gen)

integrated_node_importance = int_grad_saliency.get_node_importance(
    target_idx, class_of_interest, steps=50
)

integrated_node_importance.shape

print("integrate_node_importance.shape = {}".format(integrated_node_importance.shape))
print(
    "integrated self-importance of target node {}: {}".format(
        target_nid, integrated_node_importance[target_idx].round(2)
    )
)

G_ego = nx.ego_graph(G.to_networkx(), target_nid, radius=len(gcn.activations))

print("Number of nodes in the ego graph: {}".format(len(G_ego.nodes())))
print(
    "Number of non-zero elements in integrated_node_importance: {}".format(
        np.count_nonzero(integrated_node_importance)
    )
)

integrate_link_importance = int_grad_saliency.get_integrated_link_masks(
    target_idx, class_of_interest, steps=50
)

integrate_link_importance_dense = np.array(integrate_link_importance.todense())
print("integrate_link_importance.shape = {}".format(integrate_link_importance.shape))
print(
    "Number of non-zero elements in integrate_link_importance: {}".format(
        np.count_nonzero(integrate_link_importance.todense())
    )
)

sorted_indices = np.argsort(integrate_link_importance_dense.flatten())
N = len(graph_nodes)
integrated_link_importance_rank = [(k // N, k % N) for k in sorted_indices[::-1]]
topk = 10
# integrate_link_importance = integrate_link_importance_dense
print(
    "Top {} most important links by integrated gradients are:\n {}".format(
        topk, integrated_link_importance_rank[:topk]
    )
)

nx.set_node_attributes(G_ego, values={x[0]: {"subject": x[1]} for x in subjects.items()})

integrated_node_importance.max()

integrate_link_importance.max()

node_size_factor = 1e2
link_width_factor = 2

nodes = list(G_ego.nodes())
colors = pd.DataFrame(
    [v[1]["subject"] for v in G_ego.nodes(data=True)], index=nodes, columns=["subject"]
)
colors = np.argmax(target_encoding.transform(colors), axis=1) + 1

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
pos = nx.spring_layout(G_ego)

# Draw ego as large and red
node_sizes = [integrated_node_importance[graph_nodes.index(k)] for k in nodes]
node_shapes = ["o" if w > 0 else "d" for w in node_sizes]

positive_colors, negative_colors = [], []
positive_node_sizes, negative_node_sizes = [], []
positive_nodes, negative_nodes = [], []
node_size_scale = node_size_factor / np.max(node_sizes)
for k in range(len(nodes)):
    if nodes[k] == target_idx:
        continue
    if node_shapes[k] == "o":
        positive_colors.append(colors[k])
        positive_nodes.append(nodes[k])
        positive_node_sizes.append(node_size_scale * node_sizes[k])

    else:
        negative_colors.append(colors[k])
        negative_nodes.append(nodes[k])
        negative_node_sizes.append(node_size_scale * abs(node_sizes[k]))

# Plot the ego network with the node importances
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

# Draw the edges with the edge importances
edges = G_ego.edges()
weights = [
    integrate_link_importance[graph_nodes.index(u), graph_nodes.index(v)]
    for u, v in edges
]
edge_colors = ["red" if w > 0 else "blue" for w in weights]
weights = link_width_factor * np.abs(weights) / np.max(weights)

ec = nx.draw_networkx_edges(G_ego, pos, edge_color=edge_colors, width=weights)
plt.legend()
plt.colorbar(nc, ticks=np.arange(np.min(colors), np.max(colors) + 1))
plt.axis("off")
plt.show()

(X, _, A_index, A), _ = train_gen[0]

X_bk = deepcopy(X)
A_bk = deepcopy(A)
selected_nodes = np.array([[target_idx]], dtype="int32")
nodes = [graph_nodes.index(v) for v in G_ego.nodes()]
edges = [(graph_nodes.index(u), graph_nodes.index(v)) for u, v in G_ego.edges()]
clean_prediction = model.predict([X, selected_nodes, A_index, A]).squeeze()
predict_label = np.argmax(clean_prediction)

groud_truth_node_importance = np.zeros((N,))
for node in nodes:
    # we set all the features of the node to zero to check the ground truth node importance.
    X_perturb = deepcopy(X_bk)
    X_perturb[:, node, :] = 0
    predict_after_perturb = model.predict(
        [X_perturb, selected_nodes, A_index, A]
    ).squeeze()
    groud_truth_node_importance[node] = (
        clean_prediction[predict_label] - predict_after_perturb[predict_label]
    )

node_shapes = [
    "o" if groud_truth_node_importance[k] > 0 else "d" for k in range(len(nodes))
]
positive_colors, negative_colors = [], []
positive_node_sizes, negative_node_sizes = [], []
positive_nodes, negative_nodes = [], []
# node_size_scale is used for better visulization of nodes
node_size_scale = node_size_factor / max(groud_truth_node_importance)

for k in range(len(node_shapes)):
    if nodes[k] == target_idx:
        continue
    if node_shapes[k] == "o":
        positive_colors.append(colors[k])
        positive_nodes.append(graph_nodes[nodes[k]])
        positive_node_sizes.append(
            node_size_scale * groud_truth_node_importance[nodes[k]]
        )
    else:
        negative_colors.append(colors[k])
        negative_nodes.append(graph_nodes[nodes[k]])
        negative_node_sizes.append(
            node_size_scale * abs(groud_truth_node_importance[nodes[k]])
        )
X = deepcopy(X_bk)
groud_truth_edge_importance = np.zeros((N, N))
G_edge_indices = [(A_index[0, k, 0], A_index[0, k, 1]) for k in range(A.shape[1])]

for edge in edges:
    edge_index = G_edge_indices.index((edge[0], edge[1]))
    origin_val = A[0, edge_index]

    A[0, edge_index] = 0
    # we set the weight of a given edge to zero to check the ground truth link importance
    predict_after_perturb = model.predict([X, selected_nodes, A_index, A]).squeeze()
    groud_truth_edge_importance[edge[0], edge[1]] = (
        predict_after_perturb[predict_label] - clean_prediction[predict_label]
    ) / (0 - 1)
    A[0, edge_index] = origin_val

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
cmap = plt.get_cmap("jet", np.max(colors) - np.min(colors) + 1)
# Draw the target node as a large star colored by its true subject
nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=[target_nid],
    node_size=50 * abs(node_sizes[nodes.index(target_idx)]),
    node_color=[colors[nodes.index(target_idx)]],
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
weights = [
    groud_truth_edge_importance[graph_nodes.index(u), graph_nodes.index(v)]
    for u, v in edges
]
edge_colors = ["red" if w > 0 else "blue" for w in weights]
weights = link_width_factor * np.abs(weights) / np.max(weights)

ec = nx.draw_networkx_edges(G_ego, pos, edge_color=edge_colors, width=weights)
plt.legend()
plt.colorbar(nc, ticks=np.arange(np.min(colors), np.max(colors) + 1))
plt.axis("off")
plt.show()

