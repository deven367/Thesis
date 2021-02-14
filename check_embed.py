import numpy as np
from functions import read_json, plot_heatmap
# embeddings =  np.load('ulysses_infersent_glove.npy')
# print(embeddings[0].shape)


embeddings = read_json('gg_embeddings.jsonl')
# print(embeddings.shape)
plot_heatmap(embeddings)
