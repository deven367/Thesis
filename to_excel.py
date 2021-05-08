import pandas as pd
import numpy as np
import xlsxwriter
import pickle
import os
import scipy.io


def create_csv(embedding_path):
    for fx in os.listdir(embedding_path):
        if fx.endswith('.pkl'):
            name = fx[:-4]
            data = pickle.load(open(embedding_path+fx, 'rb'))
            write_to_sheets(data, name)
            # break
def write_to_sheets(data, name):
    writer = pd.ExcelWriter(name + '.xlsx', engine='xlsxwriter')
    for key in data[0].keys():
        df = pd.DataFrame(data[0][key])
        sheet_name = ' '.join(key.split()[-2:])
        df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
