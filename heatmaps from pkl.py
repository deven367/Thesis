import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

embedding_path = './dictionaries/sentence diff 1/'

for fx in os.listdir(embedding_path):
    if fx.endswith('.pkl'):
        name = fx[:-4]
        print("Loaded "+fx)
        fname = open(embedding_path+fx, 'rb')
        data = pickle.load(fname)
        print(data[0].keys())

        for i in range(len(alex_pope_iliad_all_books) - 1):
            # plt.figure(figsize = (15,10))
            ax = sns.heatmap(list(alex_pope_iliad_all_books[i].values()), yticklabels = list(alex_pope_iliad_all_books[i].keys()), cmap = 'hot', vmin = -1, vmax = 1)
            for j in range(len(alex_pope_iliad_all_books[i].keys())):
                ax.axhline(j, color='white', lw=2)
            title = 'Book '+str(i + 1)+' Iliad Alex Pope'
            plt.title(title)
            plt.tight_layout()
            plt.savefig(title + '.png', dpi = 300)
