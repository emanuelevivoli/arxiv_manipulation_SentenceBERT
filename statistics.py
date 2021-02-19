# In[1]:

# import all packages needed
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')

import string 

import logging
logging.basicConfig(level=logging.INFO)


# In[2]:


FILE_PATH = '../BART/'
# FILE_PATH = '../T5/'
OUT_PATH = 'statistics/'


# In[3]:


import re
regex = r"(\${1,2})(?:(?!\1)[\s\S])*\1"


# In[4]:


import json
with open(FILE_PATH + 'result.json', 'r') as f:
    documents = json.load(f)


# In[10]:


# ----------------------------------------  #
#   LIST UTILS                              #
# ----------------------------------------  #
import re

regex = r"(\${1,2})(?:(?!\1)[\s\S])*\1"

def extract_math_formula(text, regex=regex):
    logging.debug('[INIT] extract_math_formula')
    logging.debug(f"{text}")
    logging.debug('[ENDx] extract_math_formula')
    return [ele.group() for ele in re.finditer(regex, text)]

def remove_math_formula(text, regex=regex):
    logging.debug('[INIT] remove_math_formula')
    logging.debug(f"{text}")
    logging.debug('[ENDx] remove_math_formula')
    return re.sub(regex, "", text, 0, re.MULTILINE)

def clean_text(text, symbol='\n', substitute=''):
    logging.debug('[INIT] clean_text')
    logging.debug(f"{text} ")
    logging.debug('[ENDx] clean_text')
    return text.replace(symbol, substitute)


# ----------------------------------------  #
#   TOKEN UTILS                             #
# ----------------------------------------  #
def make_tokens(text):
    logging.debug('[INIT] make_tokens')
    logging.debug(f"{text} ")
    text_tokens = word_tokenize(text)
    logging.debug(f"{text_tokens} ")
    text_tokens = [word.lower() for word in text_tokens if not word in stopwords.words() and not word in string.punctuation]
    logging.debug(f"{text_tokens} ")
    logging.debug('[ENDx] make_tokens')
    return text_tokens

# ----------------------------------------  #
#   LIST UTILS                              #
# ----------------------------------------  #
def set_fuse_lists(list1, list2):
    logging.debug('[INIT] set_fuse_lists')
    logging.debug(f"{list1} --- {list2}")
    logging.debug('[ENDx] set_fuse_lists')
    return list(set(list1 + list2))

def string_fuse_list(list):
    return (" ").join(list)


# In[18]:


# ----------------------------------------  #
#   TITLE, ABSTRACT, PAPER CLASSES          #
# ----------------------------------------  #
class Title: 
    # properties
    title = ''
    title_fused_tokens = ''
    title_tokens = []
    math_formula = []

    def __init__(self, title):
        self.title = title
    
    def elaborate(self):
        logging.debug('[INIT] Title.elaborate')
        logging.debug(f"{self.title}")
        self.title = clean_text(self.title)
        self.math_formula = extract_math_formula(self.title)
        self.title_tokens = remove_math_formula(self.title)
        logging.debug(f"{self.title_tokens}")
        self.title_tokens = make_tokens(self.title_tokens)
        logging.debug(f"{self.title_tokens}")
        logging.debug(f"{self.math_formula}")
        self.title_tokens = set_fuse_lists(self.title_tokens, self.math_formula)
        self.title_fused_tokens = string_fuse_list(self.title_tokens)
        logging.debug('[ENDx] Title.elaborate')
    
    def get_tokens(self):
        return self.title_tokens

class Abstract:
    # properties
    abstract = ''
    abstract_sentences = []
    abstract_sentences_math = []
    abstract_sentences_tokens = []
    abstract_sentences_fused_tokens = []

    def __init__(self, abstract):
        self.abstract = abstract

    def elaborate(self):
        self.abstract_sentences = sent_tokenize(self.abstract)
        self.abstract_sentences_math = [extract_math_formula(text) for text in self.abstract_sentences]
        self.abstract_sentences = [remove_math_formula(text) for text in self.abstract_sentences]
        self.abstract_sentences_tokens = [make_tokens(sentence) for sentence in self.abstract_sentences]
        self.abstract_sentences_tokens = [set_fuse_lists(sentence,math) for sentence, math in zip(self.abstract_sentences_tokens, self.abstract_sentences_math)]
        self.abstract_sentences_fused_tokens = [string_fuse_list(sentence) for sentence in self.abstract_sentences_tokens]

    def get_tokens(self):
        return self.abstract_sentences_tokens


class Paper:
    # properties
    title = None
    abstract = None

    # methods
    def __init__(self, raw_paper):
        logging.debug('[INIT] Paper.__init__')
        self.title = Title(raw_paper['true_title'])
        logging.debug(f"{self.title}")
        self.title.elaborate()
        self.abstract = Abstract(raw_paper['abstract'])
        logging.debug(f"{self.abstract}")
        self.abstract.elaborate()

    def get_abstract(self):
        return self.abstract

    def get_abstract_tokens(self):
        return self.abstract.get_tokens()
    
    def get_title(self):
        return self.title

    def get_title_token(self):
        return self.title.get_tokens()


# In[42]:


# ----------------------------------------  #
#   STATISTICS UTILS                        #
# ----------------------------------------  #
def title_sentence_count(title_token, abstract_sentence_token):
    stat_token_sentence = StatisticCount()
    for base_word in title_token:
        if base_word in abstract_sentence_token:
            stat_token_sentence.add_token(base_word)
    return stat_token_sentence

def title_sentences_count(title_token, abstract_sentence_tokens):
    sentence_counts = []
    for abstract_sentence_token in abstract_sentence_tokens:
        sentence_counts.append(title_sentence_count(title_token, abstract_sentence_token))
    return sentence_counts

def title_abstract_count(title_token, abstract_sentence_tokens):
    logging.debug('[INIT] title_abstract_count')
    logging.debug(f"{abstract_sentence_tokens}")
    abstract_sentence_tokens = set().union(*abstract_sentence_tokens)
    logging.debug(f"{abstract_sentence_tokens}")
    abstract_token_list = list(abstract_sentence_tokens)
    logging.debug(f"{abstract_token_list}")
    stat_token_abstract = StatisticCount()
    for base_word in title_token:
        if base_word in abstract_token_list:
            stat_token_abstract.add_token(base_word)
    return stat_token_abstract
    logging.debug('[ENDx] title_abstract_count')

def object_list_sort_by_key(objects_to_sort, str_key):
    logging.debug('[INIT] object_list_sort_by_key')
    list_to_sort = [vars(object_to_sort) for object_to_sort in objects_to_sort]
    logging.debug(f"{list_to_sort}")
    list_to_sort = sorted(list_to_sort, key=lambda x: x.get(str_key, 0), reverse=True)
    logging.debug(f"{list_to_sort}")
    logging.debug('[ENDx] object_list_sort_by_key')
    return list_to_sort


# In[43]:


# ----------------------------------------  #
#   STATISTIC COUNT - STATISTIC CLASSES     #
# ----------------------------------------  #
class StatisticCount:
    # properties
    count = 0
    total = 1
    relative = 0
    tokens = []

    def __init__(self, stat_count=0, stat_tokens=[]):
        self.count = stat_count
        self.tokens = stat_tokens

    def add_token(self, token):
        self.count += 1
        self.total += 1
        self.relative = self.count / self.total
        self.tokens.append(token)

    def increment_total(self):
        self.relative = self.relative * self.total
        self.total += 1
        self.relative = self.relative / self.total
        


class Statistics:
    # properties
    paper = None
    abstract_counts = []
    sentence_counts = []
    sentence_counts_sort = []

    def __init__(self, paper):
        self.paper = paper

    def elaborate(self):
        logging.debug('[INIT] Statistic.elaborate')
        logging.debug(f"{self.paper}")
        title_token = self.paper.get_title_token()
        abstract_sentence_tokens = self.paper.get_abstract_tokens()
        self.sentence_counts = title_sentences_count(title_token, abstract_sentence_tokens)
        self.abstract_counts = title_abstract_count(title_token, abstract_sentence_tokens)
        self.sentence_counts_sort = object_list_sort_by_key(self.sentence_counts, 'relative')
        logging.debug('[ENDx] Statistic.elaborate')

def save_to_file(file_to_save, path_to_save = 'statistics.json'):
    with open(OUT_PATH + path_to_save, 'w') as f:
        json.dump(file_to_save, f)

def main():
    statistics = []
    lenght = len(documents)
    for i, paper in enumerate(documents):
        logging.info(f"[{i}/{lenght}] - {paper['true_title']}")
        paper_object = Paper(paper)
        statistic_object = Statistics(paper_object)
        statistic_object.elaborate()

        statistics.append(statistic_object)

    save_to_file(statistics)

if __name__ == "__main__":
    main()




# token_dictionaries_list = []
# lenght = len(documents)



# for i, paper in enumerate(documents):
    
#     # TITLE
#     print(f"[{i}/{lenght}] - {paper['true_title']}")
#     tokenized_dictionary = dict()
#     tokenized_dictionary['title'] = paper['true_title'].replace('\n', '')
#     actual_title = tokenized_dictionary['title']  # + " $$a \\text{$a$}$$"
#     reg_title = [ele.group() for ele in re.finditer(regex, actual_title)]
#     after_title = re.sub(regex, "", actual_title, 0, re.MULTILINE)
#     text_tokens = word_tokenize(after_title)
#     tokens_without_sw = [word.lower() for word in text_tokens if not word in stopwords.words() and not word in string.punctuation]
#     tokens_without_sw = list(set(tokens_without_sw + reg_title))
#     filtered_sentence = (" ").join(tokens_without_sw)
#     tokenized_dictionary['token_title'] = filtered_sentence


#     # SENTENCE
#     sentence_tokens = sent_tokenize(paper['abstract'])
#     reg_abs = []
#     after_abs = []
#     for sentence_token in sentence_tokens:
#         reg_abs.append([ele.group() for ele in re.finditer(regex, sentence_token)])
#         after_abs.append(re.sub(regex, "", sentence_token, 0, re.MULTILINE))
#     sentence_tokens = [[word.lower() for word in word_tokenize(sentence) if not word in stopwords.words() and not word in string.punctuation] for sentence in after_abs]
#     new_list = []
#     for i, sentence in enumerate(sentence_tokens):
#         new_item = sentence
#         new_item.extend(reg_abs[i])
#         new_item = list(set(new_item))
#         new_list.append(new_item)
#     abstract_without_sw = [(" ").join(sentence_without_sw) for sentence_without_sw in new_list]
#     tokenized_dictionary['splitted_abstract'] = abstract_without_sw
    

#     # STATISTICS
#     tokenized_dictionary['splitted_count'] = []
#     for sentence in new_list:
#         sentence_count = 0
#         sentence_count_list = []
#         for base_word in tokens_without_sw:
#             if base_word in sentence:
#                 sentence_count += 1
#                 sentence_count_list.append(base_word.lower())
#         tokenized_dictionary['splitted_count'].append([sentence_count, sentence_count_list])
#     token_dictionaries_list.append(tokenized_dictionary)


# A
# - delle parole che ci sono nel titolo, quante se ne trova nell'unione di tutte le sentence
# 
# B
# - per ogni sentence conto quante parole ci sono del titolo<br>
# es tit: 1 2 3 4       -> 1<br>
# es asb1: 12           -> 0.5<br>
# es asb2:              -> 0<br>
# es asb3: 34           -> 0.5<br>
# poi si ordinano, quindi mi aspetto di trovare: 
# tit, asb1, asb3, asb2 -> 1, 05, 05, 0
# 
# C
# - CSV, excell, si ordinano sulla base del primo valore, quanti ce n'è

# In[ ]:




