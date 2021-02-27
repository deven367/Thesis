import os
import pandas as pd
import numpy as np

# path = 'thesis-functions'
# cwd = os.getcwd()

embedding_path='./infersent/embeddings/fasttext/poems/'
# files = os.listdir(os.getcwd())
files = os.listdir(embedding_path)

# for f in files:
# 	print(f)
# os.mkdir(os.path.join(embedding_path,'csv' ))
# new_path = embedding_path+'csv/'
# os.chdir(new_path)
for fx in os.listdir(embedding_path):
    if fx.endswith('.npy'):
        name = fx[:-4]
        print("Loaded "+fx)
        data_item = np.load(embedding_path+fx)
        df = pd.DataFrame(data_item)

        df.to_csv(name+'.csv')
        print("CSV successfully made for {}".format(fx))
        print('---------------------------------------')
