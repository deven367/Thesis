import unidecode
import nltk
import re
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import wordnet, stopwords
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
STOPWORDS = set(stopwords.words('english'))

from collections import OrderedDict
def get_data(fname):
    with open(fname, 'r') as f:
        all = f.read()
    return all

def rm_useless_spaces(t):
    "Remove multiple spaces"
    _re_space = re.compile(' {2,}')
    return _re_space.sub(' ', t)

def make_sentences(all):
    all_cleaned = all.replace('\n',' ')
    all_cleaned = rm_useless_spaces(all_cleaned)
    all_cleaned = all_cleaned.strip()
    all_cleaned = unidecode.unidecode(all_cleaned)
    sentences = sent_tokenize(all_cleaned)
    return sentences

def write_to_file_lexical(sentences, fname):
    with open(fname[:-4]+'_lexical.txt', 'w') as f:
        for line in sentences:
            f.write(line + '\n')
    f.close()

def write_to_file_cleaned(sentences, fname):
    with open(fname[:-4]+'_cleaned.txt', 'w') as f:
        for line in sentences:
            f.write(line + '\n')
    f.close()

def remove_stopwords(text):
    sentence = []
    for word in text.split():
        if word not in STOPWORDS:
            sentence.append(word)
    return ' '.join(sentence)
