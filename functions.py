from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import json
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def newcmp(r,g,b):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(r/256, 1, N)
    vals[:, 1] = np.linspace(g/256, 1, N)
    vals[:, 2] = np.linspace(b/256, 1, N)
    newcmp = ListedColormap(vals)
    return newcmp


def read_json(path):
    with open(path, "r") as f:
        embeddings = []
        for line in f:
            embeddings.append(json.loads(line)["embeddings"])
    return embeddings

import matplotlib.pyplot as plt

def plot_heatmap(embeddings):
    sns.heatmap(cosine_similarity(embeddings, embeddings) ,square=True, cmap=newcmp, vmin=0, vmax = 1)
    inp = input('Do you have want to save the figure? ')
    if inp == 'y' or inp == 'Y':
        plt.savefig('figure.png', dpi = 300)
