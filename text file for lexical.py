import re
from functions import get_data, split_by_newline
import os
import unidecode
from collections import OrderedDict

import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import string

STOPWORDS = set(stopwords.words('english'))

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def re_patterns(text):
    # pat1 = re.compile('[\s]+')
    # doc = pat1.sub(' ', text)
    nopunc = [char for char in text if char not in string.punctuation]
    doc = ''.join(nopunc)
    # print(doc)
    doc = remove_stopwords(doc)
    # pat2 = re.compile('[^a-zA-Z0-9 ]+')
    # doc = pat2.sub('', doc)

    doc = ' '.join(list(OrderedDict.fromkeys(doc.split())))

    lemmatizer = WordNetLemmatizer()
    doc = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in doc.split()])
    return doc

def remove_stopwords(text):
    sentence = []
    for word in text.split():
        if word not in STOPWORDS:
            sentence.append(word)
    return ' '.join(sentence)


def write_to_file_from_list(sentences, fname, fpath):
    with open(fpath + '/output/' + fname[:-12]+'_pro.txt', 'w') as f:
        for sentence in sentences:
            if len(sentence) > 1:
                f.write(sentence + '\n')
    f.close()

def process(fname, fpath):
    all_data = get_data(fname, fpath)
    all_data = unidecode.unidecode(all_data)
    sentences = split_by_newline(all_data)
    clean_sentences = []

    for sentence in sentences:
        clean_sentences.append(re_patterns(sentence))

    write_to_file_from_list(clean_sentences, fname, fpath)
    print('Done processing', fname)


if __name__ == '__main__':
    fpath = './testing/'
    for fx in os.listdir(fpath):
        if fx.endswith('.txt'):
            name = fx[:-4]
            print("Loaded "+fx)
            process(fx, fpath)
            print('-'*40)
            # break
