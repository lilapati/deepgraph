import numpy as np
from matplotlib import pyplot as plt
import json


def plot_hist(history1, history2, dataset):
    fig = plt.figure(figsize=(8, 3))
    epochs = np.arange(1, 101)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history1['acc'], label='GCN')
    plt.plot(epochs, history2['acc'], label='DGCNN')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Acc')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history1['val_acc'], label='GCN')
    plt.plot(epochs, history2['val_acc'], label='DGCNN')
    plt.xlabel('Epochs')
    plt.ylabel('Val Acc')
    plt.grid(True)
    plt.legend()
    fig.suptitle(dataset.upper())
    plt.tight_layout()
    plt.savefig('gc_'+ dataset+'_acc.svg')


def main():
    history = []
    for algo, dataset in [(algo, dataset) for algo in ['gcn', 'dgcnn'] for dataset in ['MUTAG', 'PROTEINS']]:
        with open(algo + '_' +  dataset + '_history.json', 'rt') as f:
            history.append(json.load(f))

    plot_hist(history[0], history[2], 'MUTAG')
    plot_hist(history[1], history[3], 'PROTEINS')


if __name__ == '__main__':
    main()
