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
def process_v2(fname):
    all_data = get_data(fname)
    all_data = unidecode.unidecode(all_data)
    sentences = make_sentences(all_data)
    clean_sentences = []
    removed_sentences = []
    for i, sentence in enumerate(sentences):
        t = remove_punc_clean(sentence)
        if len(t) > 0:
            clean_sentences.append(t)
        else:
            removed_sentences.append(i)

    # write_to_file_lexical(clean_sentences, fname)
    print('Done processing', fname)
    return removed_sentences
