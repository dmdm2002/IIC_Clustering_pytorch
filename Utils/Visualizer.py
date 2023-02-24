import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from Utils.Option import Param


class Visualizer(Param):
    def __init__(self):
        super(Visualizer, self).__init__()

    def tSNE(self, deepfeatures, actuals, ep):
        tsne = TSNE(n_components=2, random_state=0)
        clusters = np.array(tsne.fit_transform(np.array(deepfeatures)))
        actuals = np.array(actuals)

        plt.figure()
        labels = ['Fake', 'Live']
        for i, label in zip(range(2), labels):
            idx = np.where(actuals == i)
            plt.scatter(clusters[idx, 0], clusters[idx, 1], marker='.', label=label)

        output = f'{self.OUTPUT_LOG}/visualization'
        os.makedirs(output, exist_ok=True)

        plt.savefig(f'{output}/tSNE_{ep}.png')

    def __call__(self, preds, labels, ep):
        return self.tSNE(preds, labels, ep)
