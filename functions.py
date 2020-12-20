from matplotlib.colors import ListedColormap, LinearSegmentedColormap
N = 256
def newcmp(r,g,b):
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(r/256, 1, N)
    vals[:, 1] = np.linspace(g/256, 1, N)
    vals[:, 2] = np.linspace(b/256, 1, N)
    newcmp = ListedColormap(vals)
    return newcmp

import json
def read_json(path):
    with open(path, "r") as f:
        embeddings = []
        for line in f:
            embeddings.append(json.loads(line)["embeddings"])
    return embeddings
