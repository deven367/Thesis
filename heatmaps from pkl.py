import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

embedding_path = './dictionaries/sentence diff 1/meditations/'

def plot(data, name):
    for i in range(len(data) - 1):
        plt.figure(figsize = (12,7))

        values = list(data[i].values())
        labels = list(data[i].keys())

        ax = sns.heatmap(values, yticklabels = labels, cmap = 'hot', vmin = -1, vmax = 1)

        for j in range(len(data[i].keys())):
            ax.axhline(j, color='white', lw=2)

        names = name.split()
        t = ' '.join(names[:2]).title()
        tt = names[-1].title()


#         title = 'Book '+ str(i + 1) + ' ' +tt + ' '+ t
        title = 'Book '+ str(i + 1) + ' ' + name.title()
        plt.title(title)
        plt.tight_layout()
#         plt.savefig(title + '.png', dpi = 300)
        # break
