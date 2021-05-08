#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Generate one file for all corpora to get the global dictionary,
# will only be run once.

import re
import os
from gensim import corpora

CorpusName = 'HD'
#CorpusName = 'Wiki'
print(os.getcwd())

if CorpusName == 'HD':
    years = ["speeches_{0:03d}".format(num) for num in range(97,98)]

else: # Wiki
    years = ['wiki_full']


for year in years:
    files = sorted(os.listdir('./Pipeline_output/' + year)) # original filelist
    filelist = open('./Reference/'+year+'_filelist_output.txt','w+')
    for file in files:
        filelist.write(file+'\n') # file name only: 001.txt
    filelist.close()

pat1 = re.compile('[\s]+')
pat2 = re.compile('[^a-zA-Z0-9 ]+')

total_doc = open('./final_output/corpus_'+CorpusName+'.txt', 'w+')

for year in years:
    print(year)
    file_path = './Pipeline_output/' +  year  +'/'
    filelist = open('./Reference/'+year+'_filelist_output.txt')

    cnt = 0
    for file in filelist:
        cnt += 1
        filename = file_path + file.strip('\n').split('/')[-1]
        print(filename)
        doc = pat2.sub('', ' '.join(list(open(filename))))
        doc = pat1.sub(' ', doc)
        total_doc.write(doc + '\n')

    filelist.close()
total_doc.close()

# Generate global dictionary:
Global_Dic_Path = './Reference/global_dictionary_'+CorpusName+'.pkl'

Complete_Corpora_Path = './final_output/corpus_'+CorpusName+'.txt'
full_corpora = list(open(Complete_Corpora_Path))
texts = [[word for word in document.lower().split()] for document in full_corpora]
Global_Dic = corpora.Dictionary(texts)
Global_Dic.save(Global_Dic_Path)
