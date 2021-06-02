import os
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from clean import *

from numpy import dot
from numpy.linalg import norm

meta_mi = [41, 707, 751]
hod_mi = [151, 323, 344, 466, 520, 551, 753, 783, 785, 865, 883, 890, 906, 932, 935, 954, 955, 1047, 1049, 1086, 1218, 1317, 1377, 1408, 1524, 1553, 1560, 1598, 1634, 1685, 1706, 1718,
1729, 1789, 1790, 1910, 1964, 2060, 2088, 2092, 2123, 2171, 2191, 2317, 2366, 2379, 2393, 2423]
cc_mi = [24, 82, 107, 109, 153, 244, 281, 286, 288, 295, 296, 377, 409, 492, 495, 500, 503, 659, 717, 733, 755, 756, 761, 768, 774, 776, 799, 801, 826, 828, 923, 928,
929, 938, 941, 942, 976, 1042, 1209, 1260, 1347, 1349, 1358, 1359, 1458, 1536, 1581, 1717, 1791, 1816, 1834, 1855, 1869, 1886, 1917]
pro_mi = []


def cos_sim(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(dot(a, b), norm(a)*norm(b))
    return c
    # return dot(a, b)/(norm(a)*norm(b))

def load_pmi(path):
    pmi = np.load(path)
    return pmi

def load_dictionary(path):
    fname = open(path, 'rb')
    data = pickle.load(fname)
    return data

def calc_score_v2(pmi, d, sentence_a, sentence_b):
    s = 0
    pairs = 0
    for word_1 in sentence_a.split():
        for word_2 in sentence_b.split():
            x = d.token2id.get(word_1,-1)
            y = d.token2id.get(word_2,-1)
            if x == -1 or y == -1:
                continue
            s += pmi[x][y]
            pairs += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(s, pairs)
    return c

def similarity_bwn_word_vectors(pmi, d, sentence_a, sentence_b):
    s = 0
    pairs = 0
    for word_1 in sentence_a.split():
        for word_2 in sentence_b.split():
            x = d.token2id.get(word_1,-1)
            y = d.token2id.get(word_2,-1)
            if x == -1 or y == -1:
                continue
            s += cos_sim(pmi[x], pmi[y])
            pairs += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(s, pairs)
    return c

from scipy.stats import pearsonr

def corr_coeff(pmi, d, sentence_a, sentence_b):
    s = 0
    pairs = 0
    for word_1 in sentence_a.split():
        for word_2 in sentence_b.split():
            x = d.token2id.get(word_1,-1)
            y = d.token2id.get(word_2,-1)
            if x == -1 or y == -1:
                continue
            s += pearsonr(pmi[x], pmi[y])[0]
            pairs += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(s, pairs)
    return c

def generate_lexical_vectors(pmi, d, sentences, k):
    lex_vectors = []
    for i in range(len(sentences) - k):
        lex_vectors.append(similarity_bwn_word_vectors(pmi, d, sentences[i], sentences[i+k]))
    return np.asarray(lex_vectors)


def get_all_scores(pmi, d, all_sentences, k):
    successive = []
    for i in range(len(all_sentences) - k):
        successive.append(calc_score_v2(pmi, d, all_sentences[i], all_sentences[i+k]))
    return successive

def get_all_sentences(data):
    return [line for line in data.split('\n') if len(line)>0]

def read_txt(fname):
    with open(fname, 'r') as f:
        all = f.read()
    return all

def iterator(path):
    removed_indices = []
    for fx in os.listdir(path):
        if fx.endswith('.pkl'):
            d = load_dictionary(path+fx)

        if fx.endswith('_non_norm.npy'):
            pmi = load_pmi(path+fx)

        if fx.endswith('_lexical.txt'):
            data = read_txt(path+fx)
            sentences = get_all_sentences(data)

        # if fx.endswith('_.txt'):
        #     removed_indices = process_v2(path+fx)

    print('Found pmi, dictionary, sentences and removed indices')
    return pmi, d, sentences, removed_indices

def interpolate(lex, removed_indices = []):
    for index in removed_indices:
        if index < len(lex):
            lex = np.insert(lex, index, lex[index - 1])
    return lex

if __name__ == '__main__':
    path = '../final/lexical results/the prophet/'
    parent_dir = os.path.basename(os.path.dirname(path))

    pmi, d, sentences, removed_indices = iterator(path)
    k = 1
    newlex_vectors = generate_lexical_vectors(pmi, d, sentences, k)
    lex_vectors = interpolate(newlex_vectors)
    np.save(path + parent_dir + '_cleaned_lex_vect.npy', lex_vectors)

    oldlex_vectors = get_all_scores(pmi, d, sentences, k)
    lex_vectors = interpolate(oldlex_vectors)
    np.save(path + parent_dir + '_cleaned_lexical_wt.npy', lex_vectors)
    print('Length before interpolation {}'.format(len(oldlex_vectors)))
    print('Length after interpolation {}'.format(len(lex_vectors)))
