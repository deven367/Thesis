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
