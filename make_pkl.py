import os
import numpy as np
import pickle
import string
import pathlib

pope_iliad = [218,533,686,861,1172,1367,1533,1751,1997,2152,2422,2564,2834,3006,3268,3565,3802,4003,4136,4296,4516,4744,5037]
chapman_iliad = [167,369,482,593,785,910,1020,1150,1343,1483,1718,1818,2031,2154,2337,2536,2707,2877,3000,3150,3339,3476,3768]
butler_iliad = [190,474,593,732,987,1128,1243,1392,1597,1753,1962,2075,2279,2409,2612,2850,3027,3190,3292,3437,3612,3736,3976]
lang_iliad = [157,419,541,679,840,924,1034,1154,1295,1435,1566,1673,1900,1998,2072,2247,2368,2522,2638,2773,2931,3070,3285]

pope_odyssey = [150,311,496,729,907,1014,1144,1283,1464,1689,1903,2098,2267,2437,3586,2744,2947,3078,3238,3355,3503,3632,3750]
lang_odyssey = [142,277,446,701,863,973,1078,1242,1419,1613,1796,1943,2083,2254,2445,2594,2771,2907,3073,3198,3325,3440,3550]
cowper_odyssey = [190,358,548,897,1086,1223,1352,1582,1808,2056,2314,2481,2656,2849,3086,3295,3580,3765,4032,4210,4385,4577,4727]
butler_odyssey = [111,223,367,589,718,818,902,1057,1206,1352,1501,1612,1721,1858,2007,2139,2296,2410,2552,2648,2756,2882,2987]

casaubon_meditation = [87,147,227,423,642,835,1040,1317,1520,1726,1876]
chrystal_meditation = [86,175,263,476,666,868,1079,1314,1523,1692,1855]

mackail_aeneid = [263,603,880,1183,1494,1832,2098,2355,2663,3024,3339]
dryden_aeneid = [311,677,977,1307,1678,2068,2386,2703,3096,3522,3941]
humphries_aeneid = [247,579,826,1106,1307,1670,1926,2156,2468,2829,3161]

christmas_carol = [443, 853, 1282, 1705]
great_gatsby = [355,655,1039,1407,1719,1995,2809,3077]
mysterious_affair = [363,584,792,1256,2241,2585,2909,3448,3887,4367,4804,5067]

def time_series_v3(successive, breakpoints, fname = None):
    plt_values = []
    for i in range(len(breakpoints)):

        if i == 0:
            y = successive[:breakpoints[i]]
        else:
            y = successive[breakpoints[i-1] : breakpoints[i]]

        plt_values.append(y)

    y = successive[breakpoints[-1]:]
    plt_values.append(y)

    return plt_values

def create_dict(embedding_path, k, breakpoints):
    mdict = {}
    parent_dir = os.path.basename(os.path.dirname(embedding_path))

    for book_number in range(len(breakpoints) + 1):
        sub_dict = {}
        print('Creating Book {}'.format(book_number + 1))
        for fx in os.listdir(embedding_path):
            # print(fx)
            if fx.endswith('.npy'):
                name = fx[:-4]
                # print("Loaded "+fx)

                book_name, method = get_embed_method_and_name(name)

                embed = np.load(embedding_path+fx)
                ts = time_series_v3(successive_similarities(embed, k), breakpoints)

                name = create_label_whole_book(method, parent_dir)

                sub_dict[name] = ts[book_number]

        mdict[book_number] = sub_dict
    # pickle.dump(mdict, open(parent_dir + ' ' +  book_name +'.pkl', 'wb'))
    pickle.dump(mdict, open(parent_dir +'.pkl', 'wb'))

def create_dict_whole_book(embedding_path, k):
    mdict = {}
    parent_dir = os.path.basename(os.path.dirname(embedding_path))
    sub_dict = {}
    for fx in os.listdir(embedding_path):
        if fx.endswith('.npy'):
            name = fx[:-4]
            embed = np.load(embedding_path+fx)
            book_name, method = get_embed_method_and_name(name)
            ts = successive_similarities(embed, k)

            name = create_label_whole_book(method, parent_dir)

            sub_dict[name] = ts

        if fx.endswith('_vect.npy'):
            name = fx[:-4]
            embed = np.load(embedding_path+fx)
            book_name, method = get_embed_method_and_name(name)
            # ts = successive_similarities(embed, k)

            name = create_label_whole_book(method, parent_dir)

            sub_dict[name] = embed


        if fx.endswith('_wt.npy'):
            name = fx[:-4]
            embed = np.load(embedding_path+fx)
            book_name, method = get_embed_method_and_name(name)
            # ts = successive_similarities(embed, k)

            name = create_label_whole_book(method, parent_dir)

            sub_dict[name] = embed

        if fx.endswith('_corr_ts.npy'):
            name = fx[:-4]
            embed = np.load(embedding_path+fx)
            book_name, method = get_embed_method_and_name(name)
            # ts = successive_similarities(embed, k)

            name = create_label_whole_book(method, parent_dir)
            print('Found Lex Corr', name)
            sub_dict[name] = embed


    mdict[0] = sub_dict
    pickle.dump(mdict, open(parent_dir +'_whole.pkl', 'wb'))

def successive_similarities(embeddings, k):
    successive = []
    for i in range(len(embeddings) - k):
        successive.append(cos_sim(embeddings[i], embeddings[i+k]))
    return successive

from numpy import dot
from numpy.linalg import norm

def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def label(arg):
    switcher = {
        'dcltr_base': "DeCLUTR Base",
        'dcltr_sm': "DeCLUTR Small",
        'distil': "DistilBERT",
        'if_FT': "InferSent FastText",
        'if_glove': "InferSent GloVe",
        'roberta': "RoBERTa",
        'use': "USE",
        'new_lex': 'Lexical Vectors',
        'old_lex': 'Lexical Weights',
        'lexical_wt': 'Lexical Weights',
        'lex_vect': 'Lexical Vectors',
        'lex_vect_corr_ts': 'Lexical Vectors (Corr)'
    }
    return switcher.get(arg)

def create_label_whole_book(method, parent_dir):
    # returns only the method name
    return label(method)

    # Format of Book name + Method
    # return parent_dir.title() + ' ' + label(method)

def create_label(index, method, parent_dir):
    met = label(method)
    # return 'Book ' +str(index + 1) + " " + parent_dir.title() + " " + met
    return 'Book ' +str(index + 1) + " " + met

def get_embed_method_and_name(fname):
    t = fname.split('_cleaned_')
    return  t[0].split()[-1], t[-1]

if __name__ == '__main__':
    # master_dictionary = {}
    embedding_path = '../final/casaubon meditations/'
    k = 1
    create_dict(embedding_path, k, casaubon_meditation)
    # create_dict_whole_book(embedding_path, k)
    # print(len(bl_ody))
