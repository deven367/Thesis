import numpy as np
from functions import read_json, plot_heatmap, get_cleaned_file
import os
import progressbar
bar = progressbar.ProgressBar(maxval=20,
                              widgets=[progressbar.Bar('=', '[', ']'), ' ',
                              progressbar.Percentage()])
# embeddings =  np.load('ulysses_infersent_glove.npy')
# print(embeddings[0].shape)


embeddings = read_json('gg_embeddings.jsonl')
# print(embeddings.shape)
# plot_heatmap(embeddings)

embedding_path = './datasets/'

# for fx in os.listdir(embedding_path):
#     if fx.endswith('.npy'):
#         name = fx[:-4]
#         print("Loaded "+fx)
#         embed = np.load(embedding_path+fx)
#         plot_heatmap(embed, name)
bar.start()
fname = 'Republic By Plato.txt'
get_cleaned_file(embedding_path+fname)
bar.finish()
