import numpy as np
import os

def fix(embedding_path):
    for fx in os.listdir(embedding_path):
        if fx.endswith('.npy'):
            name = fx[:-4]
            embed = np.load(embedding_path+fx)
            print("Loaded "+fx)

            print("Old Length {}". format(len(embed)))

            new_embed = embed[:-1]
            print("New Length {}". format(len(new_embed)))

            np.save(fx, new_embed)


if __name__ == '__main__':
    embedding_path = '../final/mackail/'
    fix(embedding_path)
