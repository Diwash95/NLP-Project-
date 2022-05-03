#!/usr/bin/env python
# coding: utf-8

# ##References
# [1] Z. Zhang, H. Zhao, and R. Wang, “Machine Reading Comprehension: The Role of Contextualized Language Models and Beyond,” CoRR, vol. abs/2005.06249, 2020, [Online]. Available: https://arxiv.org/abs/2005.06249 
# 
# [2] Y. Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” CoRR, vol. abs/1907.11692, 2019, [Online]. Available: http://arxiv.org/abs/1907.11692 
# 
# [3] J. Ni, T. Young, V. Pandelea, F. Xue, V. Adiga, and E. Cambria, “Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey,” CoRR, vol. abs/2105.04387, 2021, [Online]. Available: https://arxiv.org/abs/2105.04387
# 
# [4] P. Dwivedi, “NLP - building a question answering model,” Medium, 11-Jul-2018. [Online]. Available: https://towardsdatascience.com/nlp-building-a-question-answering-model-ed0529a68c54. [Accessed: 15-Apr-2022].
# 
# [5] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+ Questions for Machine Comprehension of Text,” arXiv e-prints, p. earXiv:1606.05250, 2016.
# 
# [6] https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626
# 
# [7]https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f

# In[2]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install transformers')


# In[3]:


from datasets import load_dataset
dataset = load_dataset('squad_v2', split='train')


# In[4]:


print(dataset.shape)
print(dataset.column_names)
dataset.features


# # test distilBERT

# In[5]:


#import libraries
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch


# In[6]:


#define the distilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")


# In[7]:


#run the modeel on the dataset to get preds
preds = []
for qa in dataset:
  inputs = tokenizer(qa['question'], qa['context'], return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs)
  answer_start_index = outputs.start_logits.argmax()
  answer_end_index = outputs.end_logits.argmax()
  predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
  x = tokenizer.decode(predict_answer_tokens)
  preds.append(x)


# In[ ]:


#functions to compute f1 score and exact match score
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

