import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import pandas as pd

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot(data, name):
    for i in range(len(data)):
        plt.figure()

        values = list(data[i].values())
        labels = list(data[i].keys())

        ax = sns.heatmap(values, yticklabels = labels, cmap = 'hot', vmin = -1, vmax = 1)
        '''
            code to check whether all strips are of same length
        '''
        np_values = np.asarray(values)
        for val, lab in zip(np_values, labels):
            print('{} Length is {}'.format(lab, len(val)))

        # normalize each strip separately, if null, put zero
        norm_ = []
        for val in values:
            if np.count_nonzero(np.isnan(val)) > 0:
                val = np.nan_to_num(val, nan=0)
            norm_.append(normalize(val))


        for j in range(len(data[i].keys())):
            ax.axhline(j, color='white', lw=2)

        # have the name of the pkl file in the format FN LN BN
        names = name.split()
        first_name = names[0]
        last_name = names[1]
        book_name = names[-1]

        # format for Book number Last Name Book name
        title = 'Book '+ str(i + 1) + ' ' + last_name.title() + ' ' + book_name.title()

        # format for strip plot of the whole book
        # title = name.title()

        # print(title)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(title + '.png', dpi = 300)
        print("Created {}".format(title))
        # break

def iterator(embedding_path):
    for fx in os.listdir(embedding_path):
        if fx.endswith('.pkl'):
            name = fx[:-4]
            print("Loaded "+fx)
            fname = open(embedding_path+fx, 'rb')
            data = pickle.load(fname)
            plot(data, name)
            #print(data[0].values())
            # break

if __name__ == '__main__':
    embedding_path = './'
    iterator(embedding_path)
