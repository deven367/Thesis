from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
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
    return np.asarray(embeddings)

import matplotlib.pyplot as plt

def plot_heatmap(embeddings, fname):
    cmp = newcmp(0,0,256)
    sns_plot = sns.heatmap(cosine_similarity(embeddings, embeddings) ,square=True, cmap=cmp, vmin=0, vmax = 1)
    # plt.show()
    # inp = input('Do you have want to save the figure? ')
    # if inp == 'y' or inp == 'Y':
    plt.savefig(fname+'.png', dpi = 300)
    plt.clf()

def get_data(fname):
    with open(fname, 'r') as f:
        all = f.read()
    return all

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
def make_sentences(all):
    all_cleaned = all.replace('\n',' ')
    sentences = sent_tokenize(all_cleaned)
    return sentences

import string
def only_words(text):
    c = 0
    for word in text:
        if word not in string.punctuation:
            c += 1
    return c

from nltk.corpus import stopwords
def num_stopwords(text):
    c = 0
    for word in text:
        if word in stopwords.words('english'):
            c += 1
    return c

def make_dataframe(sentences):
    df = pd.DataFrame(sentences, columns=['text'])
    df['index'] = [i for i in range(len(df))]
    df['length'] = df['text'].apply(len)
    df['tokens'] = df['text'].apply(word_tokenize)
    df['n_tokens'] = df['tokens'].apply(len)
    df['only_words'] = df['tokens'].apply(only_words)
    df['stopwords'] = df['tokens'].apply(num_stopwords)
    df['valid'] = df['only_words'] - df['stopwords']
    df2 = df[ (df['valid'] >= 2) ]
    df3 = df[ (df['stopwords'] ==0) & (df['valid'] == 1)]
    cleaned_df = pd.concat([df2,df3])
    cleaned_df.sort_index(inplace = True)
    return cleaned_df
