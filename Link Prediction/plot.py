import numpy as np
from matplotlib import pyplot as plt
import json


def plot_hist(history1, history2, history3, history4, dataset):
    fig = plt.figure(figsize=(8, 3))
    epochs = np.arange(1, 101)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history1['acc'], label='Attri2Vec')
    # plt.plot(epochs, history2['acc'], label='GCN')
    plt.plot(epochs, history3['acc'], label='GraphSage')
    plt.plot(epochs, history4['acc'], label='HinSage')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Acc')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history1['val_acc'], label='Attri2Vec')
    # plt.plot(epochs, history2['val_acc'], label='GCN')
    plt.plot(epochs, history3['val_acc'], label='GraphSage')
    plt.plot(epochs, history4['val_acc'], label='HinSage')
    plt.xlabel('Epochs')
    plt.ylabel('Val Acc')
    plt.grid(True)
    plt.legend()
    fig.suptitle(dataset.upper())
    plt.tight_layout()
    plt.savefig('lp_'+ dataset+'_acc.svg')


def main():
    history = []
    for i, (algo, dataset) in enumerate([(algo, dataset) for algo in ['attri2vec', 'gcn', 'graphsage', 'hinsage'] for dataset in ['citeseer', 'cora', 'pubmed']]):
        with open(algo + '_' +  dataset + '_history.json', 'rt') as f:
            history.append(json.load(f))

    plot_hist(history[0], history[3], history[6], history[9], 'citeseer')
    plot_hist(history[1], history[4], history[7], history[10], 'cora')
    plot_hist(history[2], history[5], history[8], history[11], 'pubmed')



if __name__ == '__main__':
    main()
