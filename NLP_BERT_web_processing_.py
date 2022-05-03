#!/usr/bin/env python
# coding: utf-8

# #Import Statements

# In[56]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install transformers')


# In[57]:


# get transformers then bert along with other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import torch

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from urllib.request import urlopen
from bs4 import BeautifulSoup


# # Web Processing

# In[58]:


# Code to convert all the website information into a text data 

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
user_agent_list = [
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
]
headers={'User-Agent':user_agent,} 

#Removes all the script text when data extracted from a page 
def remove_scripts(page):
  data=BeautifulSoup(page)
  for i in data(['script','style']):
    i.decompose()
  return ' '.join(data.stripped_strings).lower()

#Gets the url and reads the contents
def page_data(url):
  req=urllib.request.Request(url,None,headers)
  res=urllib.request.urlopen(req)
  return res.read()

#Gets the data page request 
def urlFetch(sample_url):
  raw_data=page_data(sample_url)
  text=remove_scripts(raw_data)
  return text


# # Bert Model

# In[59]:


# Define the bert tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Load the fine-tuned model for question answering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model.eval()


# In[60]:


#preprocessing text information 
def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


# In[61]:


#predict answer based on the question
def ans_pred(context,query):
  context = normalize_text(context)
  inputs = tokenizer.encode_plus(query, context, return_tensors='pt')
  outputs = model(**inputs)
  answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
  answer_end = torch.argmax(outputs[1]) + 1 
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
  return answer


# In[62]:


#Computer prediction
def give_answer(context,query,predval):
  prediction = ans_pred(context,query)
  predval.append(prediction) 
  print(f"Question: {query}")
  print(f"Prediction: {prediction}")
  print("\n")


# In[63]:


# Singnificance of using BERT Model 
# we are spliting the data into 510 as a max bucket size is 512 in one container

text=urlFetch("https://en.wikipedia.org/wiki/Apollo_16") # Wikipedia of Apollo_16 mission
print(len(text))
n = 510
chunks = [text[i:i+n] for i in range(0, len(text), n)] # Holds 510 words per list (creates a list of list)


# In[65]:


question="What is the best time?"


# In[79]:


#Analyzing question and answer
qan=[]
id=np.random.randint(0,len(chunks))
context=chunks[id]
give_answer(context,question,qan)


# In[ ]:




