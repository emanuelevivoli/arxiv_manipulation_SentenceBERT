#!/usr/bin/env python
# coding: utf-8

# In[8]:


FILE_PATH = '../BART/'
# FILE_PATH = '../T5/'


# In[9]:


import json


with open(FILE_PATH + 'result.json', 'r') as f:
    documents = json.load(f)

print(documents)


# In[15]:


sentences = []

for paper in documents:
    sentences.append(paper['abstract'])

print(sentences)


# # Start Now with Sentence Transformer on sentences (abstract) to see which one is more similar 

# In[24]:


from sentence_transformers import SentenceTransformer, util


# In[12]:


model = SentenceTransformer('paraphrase-distilroberta-base-v1')


# In[3]:


#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']


# In[16]:


#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)


# In[22]:


len(sentence_embeddings)


# In[25]:


cos_sim = dict()

for i in range(len(sentence_embeddings)):
    
    for j in range(i, len(sentence_embeddings)):

        cos_sim_i_j = util.pytorch_cos_sim(sentence_embeddings[i], sentence_embeddings[j])

        cos_sim[f"{i}_{j}"] = cos_sim_i_j
        cos_sim[f"{j}_{i}"] = cos_sim_i_j

        print(f"Cosine-Similarity {i}_{j} and ({j}_{i}):", cos_sim_i_j)


# In[26]:


cos_sim


# In[ ]:




