#!/usr/bin/env python
# coding: utf-8

# In[48]:


import nltk
from nltk.corpus import stopwords
import string 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize

import logging
logging.basicConfig(level=logging.INFO)


# In[49]:


FILE_PATH = '../BART/'
# FILE_PATH = '../T5/'
OUT_PATH = 'statistics/'


# In[50]:


import re
regex = r"(\${1,2})(?:(?!\1)[\s\S])*\1"
# test_str = "This is an example $$a \\text{$a$}$$. How to remove it? Another random math expression $\\mathbb{R}$..."
# subst = "MATH"
# result = re.sub(regex, subst, test_str, 0, re.MULTILINE)


# In[51]:


import json
with open(FILE_PATH + 'result.json', 'r') as f:
    documents = json.load(f)


# In[52]:


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


# In[53]:


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
        logging.debug(f"{self.title_tokens}")
        self.title_fused_tokens = string_fuse_list(self.title_tokens)
        logging.debug('[ENDx] Title.elaborate')
    
    def get_tokens(self):
        return self.title_tokens

    def get_title(self):
        return self.title

    def __str__(self):
        return f"Title \n\t title={self.title} \n\t title_fused_tokens={self.title_fused_tokens} \n\t title_tokens={self.title_tokens} \n\t math_formula={self.math_formula} "

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
        logging.debug('[INIT] Abstract.elaborate')
        self.abstract_sentences = sent_tokenize(self.abstract)
        self.abstract_sentences_math = [extract_math_formula(text) for text in self.abstract_sentences]
        self.abstract_sentences = [remove_math_formula(text) for text in self.abstract_sentences]
        self.abstract_sentences_tokens = [make_tokens(sentence) for sentence in self.abstract_sentences]
        self.abstract_sentences_tokens = [set_fuse_lists(sentence,math) for sentence, math in zip(self.abstract_sentences_tokens, self.abstract_sentences_math)]
        self.abstract_sentences_fused_tokens = [string_fuse_list(sentence) for sentence in self.abstract_sentences_tokens]
        logging.debug('[ENDx] Abstract.elaborate')

    def get_tokens(self):
        return self.abstract_sentences_tokens

    def __str__(self):
        return f"Abstract \n\t abstract={self.abstract} \n\t abstract_sentences={self.abstract_sentences} \n\t abstract_sentences_math={self.abstract_sentences_math} \n\t abstract_sentences_tokens={self.abstract_sentences_tokens} \n\t abstract_sentences_fused_tokens={self.abstract_sentences_fused_tokens}"


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

    def get_title_title(self):
        return self.title.get_title()

    def get_title_token(self):
        return self.title.get_tokens()

    def __str__(self):
        return f"Paper\n {self.title}\n {self.abstract}"


# In[54]:


# ----------------------------------------  #
#   STATISTICS UTILS                        #
# ----------------------------------------  #
def title_sentence_count(title_token, abstract_sentence_token):
    stat_token_sentence = StatisticCount()
    for base_word in title_token:
        if base_word in abstract_sentence_token:
            stat_token_sentence.add_token(base_word)
        else:
            stat_token_sentence.increment_total()
    return stat_token_sentence

def title_sentences_count(title_token, abstract_sentence_tokens):
    logging.debug('[INIT] title_sentences_count')
    logging.debug(f"[title] - {title_token}")
    logging.debug(f"[tokens] - {abstract_sentence_tokens}")
    sentence_counts = []
    for i, abstract_sentence_token in enumerate(abstract_sentence_tokens):
        logging.debug(f"[{i}] --- title    : {title_token}")
        logging.debug(f"[{i}] --- sentence : {abstract_sentence_token}")
        stat_sentence_count = title_sentence_count(title_token, abstract_sentence_token)
        logging.debug(f"[{i}] --- intersect: {stat_sentence_count}")
        sentence_counts.append(stat_sentence_count)
    logging.debug(f"[sentence_counts] - {sentence_counts[0]}")
    logging.debug('[INIT] title_sentences_count')
    return sentence_counts

def title_abstract_count(title_token, abstract_sentence_tokens):
    logging.debug('[INIT] title_abstract_count')
    logging.debug(f"[title] - {title_token}")
    logging.debug(f"{abstract_sentence_tokens}")
    abstract_sentence_tokens = set().union(*abstract_sentence_tokens)
    logging.debug(f"{abstract_sentence_tokens}")
    abstract_token_list = list(abstract_sentence_tokens)
    logging.debug(f"{abstract_token_list}")
    stat_token_abstract = StatisticCount()
    for base_word in title_token:
        if base_word in abstract_token_list:
            stat_token_abstract.add_token(base_word)
        else:
            stat_token_abstract.increment_total()
    logging.debug('[ENDx] title_abstract_count')
    return stat_token_abstract
    

def object_list_sort_by_key(objects_to_sort, str_key):
    logging.debug('[INIT] object_list_sort_by_key')
    logging.debug(f"{objects_to_sort}")
    list_to_sort = [vars(object_to_sort) for object_to_sort in objects_to_sort]
    logging.debug(f"{list_to_sort}")
    sorted_list = sorted(list_to_sort, key=lambda x: x.get(str_key, 0), reverse=True)
    logging.debug(f"{sorted_list}")
    sorted_objects = [StatisticCount(sorted_object) for sorted_object in sorted_list]
    logging.debug('[ENDx] object_list_sort_by_key')
    return sorted_objects


# In[55]:


# ----------------------------------------  #
#   STATISTIC COUNT - STATISTIC CLASSES     #
# ----------------------------------------  #
class StatisticCount:
    # properties
    count = 0
    total = 0
    relative = 0
    tokens = []

    def __init__(self, kwargs={}):
        if kwargs:
            self.count = kwargs['count']
            self.total = kwargs['total']
            self.relative = kwargs['relative']
            self.tokens = kwargs['tokens']
        else:
            self.count = 0
            self.total = 0
            self.relative = 0
            self.tokens = []

    
    def add_token(self, token):
        self.count += 1
        self.total += 1
        self.relative = self.count / self.total
        self.tokens.append(token)

    def increment_total(self):
        self.relative = self.relative * self.total
        self.total += 1
        self.relative = self.relative / self.total
        
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"\nStatisticCount\n count={self.count}\n total={self.total}\n relative={self.relative}\n tokens={self.tokens}"


class PaperStatistics:
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

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"\nPaperStatistics\n paper={self.paper}\n abstract_counts={self.abstract_counts}\n sentence_counts={self.sentence_counts}\n sentence_counts_sort={self.sentence_counts_sort}"


# In[56]:


def calculate_statistics():
    statistics = []
    lenght = len(documents)
    for i, paper in enumerate(documents):
        paper_object = Paper(paper)
        logging.info(f"[{i}/{lenght - 1}] - {paper_object.get_title_title()}")

        statistic_object = PaperStatistics(paper_object)
        statistic_object.elaborate()

        statistics.append(statistic_object)
    return statistics

def calculate_one_statistic(index):
    statistics = []
    
    paper = documents[index]
    paper_object = Paper(paper)
    logging.info(f"[{index}] - {paper_object.get_title_title()}")
    statistic_object = PaperStatistics(paper_object)
    statistic_object.elaborate()

    statistics.append(statistic_object)
    return statistics

statistics = calculate_statistics()
# statistics = calculate_one_statistic(0)


# import splitter
# 
# print(splitter.split('extendedlagrangian'))
# print(splitter.split('appearadditional'))
# print(splitter.split('additionalself-interaction'))

# import enchant
# print(enchant.list_languages())
# 
# from cwsplit import load_dict
# load_dict('en_en')
# 
# print(split('blackboard', 'en_en'))

# for i, paper in enumerate(documents):
#     
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
# 
# 
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
#     
# 
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

# In[57]:


vars_statistics = [vars(stat) for stat in statistics]


# In[58]:


vars(vars_statistics[0]['sentence_counts'][0])


# with open(OUT_PATH + 'statistics.json', 'w') as f:
#     json.dump([vars(stat) for stat in statistics], f)

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

# - title.title
# - title.title_tokens
# - abstract_counts
# - sentence_counts_sort

# In[69]:


def trp(l, n):
    return l[:n] + ['-']*(n-len(l))


# In[73]:


class CSVPaperStatistic:
    def __init__(self, statistic):
        # title
        self.title = statistic.paper.get_title_title()
        self.title_tokens = statistic.paper.get_title_token()
        self.title_total = len(self.title_tokens)

        logging.debug(self.title, self.title_tokens, self.title_total)

        # abstract
        self.abstract_count = statistic.abstract_counts.count
        self.abstract_relative = statistic.abstract_counts.relative
        self.abstract_tokens = statistic.abstract_counts.tokens

        logging.debug(self.abstract_count, self.abstract_relative, self.abstract_tokens )

        # sentences
        self.sent_counts = [ stat_sentence.relative for stat_sentence in statistic.sentence_counts_sort ]

        logging.debug(trp(self.sent_counts, 10))


    def get_list(self):
        return [ self.title, self.title_tokens, self.title_total, self.abstract_count, self.abstract_relative, self.abstract_tokens ] + trp(self.sent_counts, 10)
        


# In[74]:


def lists_from_statistics(statistics):
    to_csv_statistics = []
    for paper_statistic in statistics:
        csv_paper_statistic = CSVPaperStatistic(paper_statistic)
        csv_list = csv_paper_statistic.get_list()
        print(csv_list)
        to_csv_statistics.append(csv_list)
    return to_csv_statistics


# In[75]:


import csv

fields = [  'title', 'title_tokens', 'title_total', 'abstract_count', 'abstract_relative', 'abstract_tokens', 
            'sent_0_count', 'sent_1_count', 'sent_2_count', 'sent_3_count', 'sent_4_count', 
            'sent_5_count', 'sent_6_count', 'sent_7_count', 'sent_8_count', 'sent_9_count']

myData = lists_from_statistics(statistics)
myData.insert(0, fields)

myFile = open(OUT_PATH + 'statistics.csv', 'w')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(myData)


# In[ ]:




