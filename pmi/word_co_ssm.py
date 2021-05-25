#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (1. Generating global topics --> Gen_global_topics.py)
# 2. Generating SSM (this code) (then 1. segmentation by Matlab 2. LDA (filelist needed))
# (3. Segment Embeddings --> GlobalEmbedding.py)

from gensim import corpora
import itertools
import os, pickle, re
import sys,time
import numpy as np
from scipy import io as sio
from scipy import sparse as ssparse
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def process_document(args):
    """
    Process a document for word count, word pair count at sentence level.
    """
    # Unpacking arguments
    corpus, file, dictionary = args
    n_token = len(dictionary)
    filepath = './Pipeline_output/' + corpus + '/' + file
    filehandle = open(filepath,'r')

    try:
        Sentences = list(filehandle)
        filehandle.close()
    except :
        print('Corrupted file: ' + filepath + ', exists')
        return False
    if len(Sentences) == 0:
        return False

    data,indices,indptr = [], [], [0]
    wordpair=[]
    wp_cnt, row_ind, col_ind = [],[],[]
    for sentence in Sentences:
        idx = list(set(dictionary[word] for word in re.findall('[a-zA-Z]+', sentence) \
                       if word in dictionary.keys()))
        #We don't care about duplicated words in the same sentence
        indices += idx
        data += [1] * len(idx)

        #We need a symmetrical matrix with 0 in the diagonal
        wordpair += list(itertools.permutations(idx, 2))
        indptr.append(len(indices))

    # document matrix [Word X Sentence] , each sentence is represented as a row vector of the
    # size of vocabulay, where 1 indicates a word exists in this sentence.
    # the sum of all row vectors represents total word count
    doc_word_sentence_matrix = ssparse.csr_matrix((data, indices, indptr),
                                   shape=(len(Sentences), n_token),
                                   dtype=int)

    # initialize word pair csr matrix in the form of
    # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    # where data, row_ind and col_ind satisfy the relationship:
    # a[row_ind[k], col_ind[k]] = data[k].
    wp_cnt = [1]*len(wordpair)

    row_ind ,col_ind = map(list,zip(*wordpair))
    doc_wp_matrix = ssparse.csr_matrix((wp_cnt, (row_ind, col_ind)),
                                   shape=(n_token, n_token),
                                   dtype=int)    

    return (doc_word_sentence_matrix, doc_wp_matrix)

def gen_ssm_iter(args):
    """
    This funtion generates Sentence Similarity Matrix (SSM) for a document
    Using PMI_pos_norm as *PMI*
    """
    # args=(corpus, file, doc_wc_dic[file], PMI_pos_norm)
    # Unpacking arguments
    corpus, file, doc_word_sentence_matrix, PMI = args

    #doc_word_sentence_matrix = ssparse.load_npz(matrix_path + file[:-4] + 'word_matrix.npz')#.todense()
    num_sentences_doc = doc_word_sentence_matrix.shape[0]
    ssm = np.zeros((num_sentences_doc, num_sentences_doc))
    coo = doc_word_sentence_matrix.nonzero()
    sen_wordidx_dic= {x:[] for x in range(num_sentences_doc)}
    for (x,y) in zip(coo[0], coo[1]):
        sen_wordidx_dic[x].append(y)

    for i,j in itertools.combinations(range(num_sentences_doc),2):
        sij = itertools.product(sen_wordidx_dic[i], sen_wordidx_dic[j])
        if len(sen_wordidx_dic[i])*len(sen_wordidx_dic[j])>0: #both not empty
            score = sum([PMI[x,y] for (x,y) in sij]) / (len(sen_wordidx_dic[i]) * len(sen_wordidx_dic[j]))
            ssm[i,j] = score
            ssm[j,i] = score
    sio.savemat(matrix_path + "SSM_" + file[:-4] + ".mat", {"SSM":ssm})
    return ssm

def append_and_remove(small, other):
    """ Append small file to another one and remove small file
    input
        small: File path to small file
        other: File to append to, if not present, we simply remove the small file
    """
    try:
        with open(small,'r') as small_handle:
            small_content = small_handle.read()
    except FileNotFoundError as fileerr:
        print('Small file not present', fileerr)
        return

    if os.path.isfile(other):
        # Only append if other file exists
        with open(other,'a') as other_handle:
            other_handle.write(small_content)

    try:
        os.remove(small)
    except FileNotFoundError as fileerr:
        print('File not present', fileerr)
        return



#%%
if __name__ == '__main__':
    #CorpusName = 'IJCNN' 
    CorpusName = 'HD'
    
    if CorpusName == 'HD':
        corpus = 'speeches_{0:03d}'.format(int(sys.argv[1]))
    else: 
        corpus = str(int(sys.argv[1])+1998) # format for IJCNN data 
    # (slurm array has a range limit of 1000, so I substracted 1998)
    

    #corpus = 'speeches_097'
    #corpus ='test'
    #corpus ='speeches_{0:03d}'.format(97)
    
    #Load/ Generate global dictionary:
    Global_Dic_Path = './Reference/global_dictionary_'+CorpusName+'.pkl'
    
    # Generate Complete Corpora by mergetext_full.py
    #Complete_Corpora_Path = './final_output/corpus_'+CorpusName+'.txt'
    
#    if os.path.isfile(Global_Dic_Path):
    Global_Dic = corpora.Dictionary.load(Global_Dic_Path)

    dictionary = Global_Dic.token2id
    n_token = len(dictionary)

    
    corpus_path = './Pipeline_output/' + corpus + '/'
    doc_list = sorted([
        file for file in os.listdir(corpus_path) \
            if file.endswith('.txt') \
                and ((CorpusName == 'HD' and file[-9:-4].isdigit()) # for HD data
                or CorpusName == 'IJCNN')
            ]
        )
    doc_wc_dic = {}
    doc_wp_dic = {}

    num_sentence = 0
    wordcount = np.zeros([1, n_token], dtype=int)
    wordpaircount = np.zeros([n_token,n_token], dtype=int)
    cnt,removed = 0, 0
    t0 = time.time()
    print('Begin Preprocessing')

    small_size = 100 #Threshold to concatenate
    for file in doc_list:
        # Processing file in sorted order
        res = process_document((corpus, file, dictionary))
        #if res == False or (CorpusName == 'HD' and res[0].shape[0] <= small_size): 
        if res == False :
            # only HD data needs to be taken care of small files because files are artificially splitted
            removed += 1
            continue
        elif CorpusName == 'HD' and res[0].shape[0] <= small_size:
            removed += 1
            # if cannot process current file or find last partition, word count and word pair count from
            # this document is ignored
            last_file = file[:-9] + "{0:05d}".format(int(file[-9:-4]) -1) + '.txt'

            # apppend current file to last file and remove current file
            # if last_file not present, just remove current file
            append_and_remove(corpus_path + file, corpus_path + last_file)
            #if res != False and last_file in doc_wc_dic.keys():
            if last_file in doc_wc_dic.keys():
                # Append current result to last one if available
                # words in sentence needs to be concatenated
                doc_wc_dic[last_file] = ssparse.vstack([doc_wc_dic[last_file], res[0]])
                doc_wp_dic[last_file] = doc_wp_dic[last_file] + res[1]
#            else: # word count & wp count ignored
#                continue

        else:
            doc_wc_dic[file] = res[0]
            doc_wp_dic[file] = res[1]
            cnt += 1

        wordcount += res[0].sum(axis=0)
        wordpaircount += res[1]
        num_sentence += res[0].shape[0]

        if cnt % 500 == 0 or cnt == len(doc_list) - removed:
            print("Finished Preprocessing for {0:d} files, took {1:0.2f} seconds."\
                  .format(cnt, time.time() - t0))

    output_path = './final_output/'+corpus + '/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(output_path + 'doc_wordcount_dic.pkl', 'wb') as filehandle:
        pickle.dump(doc_wc_dic,filehandle)

    with open(output_path + 'doc_wordpair_dic.pkl', 'wb') as filehandle:
        pickle.dump(doc_wp_dic, filehandle)

    wordcount_product = np.outer(np.transpose(wordcount), wordcount)

    np.save(output_path + 'wordprob.npz', wordcount/num_sentence)



    # Pointwise Mutual Information PMI(x,y) = log(p(x,y)/p(x)/p(y))
    with np.errstate(divide='ignore', invalid='ignore'):
        #adds in a value of 1 to count
        PMI = np.log((wordpaircount * num_sentence +1) / (wordcount_product+1))
    """
    from scipy.io import savemat
    savemat(output_path +'PMI.mat', {'PMI': PMI})
    """
    # Only keep positive info
    PMI_pos = np.maximum(
            np.nan_to_num(PMI, nan=0.0, neginf=0.0),
            # nan_to_num requires numpy version of 1.17 and above
            np.zeros(PMI.shape, dtype=float),
            )

    # Normalize 
    # can't use MinMaxScaler as it normalizes by features/samples
    # scaler = MinMaxScaler()
    # PMI_norm = scaler.fit_transform(PMI)
    PMI_norm = (PMI-np.min(PMI)) / (np.max(PMI)-np.min(PMI))
    np.fill_diagonal(PMI_norm,1)
    
    
    np.save(output_path + 'wordcount.npz', wordcount)
    np.save(output_path + 'wordprob.npz', wordcount/num_sentence)
    np.save(output_path + 'wordpaircount.npz', wordpaircount)
    np.save(output_path + 'PMI.npz', PMI)
    #np.save(output_path + 'PMI_pos.npz', PMI_pos)
    np.save(output_path + 'PMI_norm.npz', PMI_norm)
    
    """
    np.savetxt(output_path + 'wordcount.csv', wordcount)
    np.savetxt(output_path + 'wordpaircount.csv', wordpaircount)
    np.savetxt(output_path + 'PMI.csv', PMI, fmt='%.4f')
    np.savetxt(output_path + 'PMI_pos.csv', PMI_pos, fmt='%.4f')
    np.savetxt(output_path + 'PMI_pos_norm.csv', PMI_pos_norm, fmt='%.4f')
    """
    
    print('Begin Generating SSM')
    cnt = 0
    t0 = time.time()
    matrix_path = './final_output/' + corpus + '/SSM/'
    Path(matrix_path).mkdir(parents=True, exist_ok=True)
    SSM_dic = {}
    for file, doc_ws in doc_wc_dic.items():
        ssm = gen_ssm_iter((corpus, file, doc_ws, PMI_norm))
        cnt += 1
        if cnt % 500 == 0 or cnt == len(doc_wc_dic.items()):
            print("Finished Generating SSM for {0:d} files, took {1:0.2f} seconds."\
                  .format(cnt, time.time() - t0))

    # Generate updated filelist
    files = sorted(os.listdir(matrix_path))
    filelist = open('./final_output/'+corpus+'/filelist_ssm.txt','w+')
    for file in files:
        #f_filelist.write('./Pipeline_input/'+years[i]+'/'+file+'\n') # with directory: ./Pipeline_input/test1/001.txt
        filelist.write(file+'\n') # file name only: 001.txt
    filelist.close()
