import os
import numpy as np
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from make_pkl import *
from functions import normalize

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


def sections(embedding_path, breakpoints):
    parent_dir = os.path.basename(os.path.dirname(embedding_path))
    for fx in os.listdir(embedding_path):
        if fx.endswith('.npy'):
            name = fx[:-4]
            embed = np.load(embedding_path+fx)
            book_name, method = get_embed_method_and_name(name)
            split_sections(embed, breakpoints, parent_dir, method)


def split_sections(embed, breakpoints, name, method):
    n = len(breakpoints) + 1
    l = label(method)
    if 'et al' in name:
        titles = [f'Book {i+1} Lang et al. {l}' for i in range(n)]
        new_path = 'Lang et al. ' + l
    else:
        titles = [f'Book {i+1} {name.title()} {l}' for i in range(n)]
        new_path = name.title() + ' ' +   l

    os.makedirs(new_path, exist_ok = True)
    for i in range(len(breakpoints)):
        if i == 0:
            sect = embed[:breakpoints[i]]
            norm = normalize(cosine_similarity(sect, sect))
            plot_sections(norm, titles[i], new_path)

        else:
            sect = embed[breakpoints[i-1] : breakpoints[i]]
            norm = normalize(cosine_similarity(sect, sect))
            plot_sections(norm, titles[i], new_path)
        print(f'Plotted {titles[i]}')

    sect = embed[breakpoints[-1] : ]
    norm = normalize(cosine_similarity(sect, sect))
    plot_sections(norm, titles[-1], new_path)
    print(f'Plotted {titles[-1]}')

def plot_sections(norm, title, new_path):
    p = Path(new_path)
    plt.title(title)
    sns.heatmap(norm, cmap = 'hot', vmin = 0, vmax = 1, square = True, xticklabels = 50, yticklabels = 50)
    plt.savefig(p/f'{title}.png', dpi = 300, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    path = '../final/casaubon meditations/'
    sections(path, casaubon_meditation)
