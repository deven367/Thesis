import numpy as np
from functions import read_json, plot_heatmap, get_cleaned_file
import os

# embeddings =  np.load('ulysses_infersent_glove.npy')
# print(embeddings[0].shape)


# embeddings = read_json('gg_embeddings.jsonl')
# print(embeddings.shape)
# plot_heatmap(embeddings)

embedding_path = r'C:/Users/Deven Mistry/Downloads/Untitled folder/distil/'

for fx in os.listdir(embedding_path):
    if fx.endswith('.npy'):
        name = fx[:-4]
        print("Loaded "+fx)
        embed = np.load(embedding_path+fx)
        plot_heatmap(embed, name)
