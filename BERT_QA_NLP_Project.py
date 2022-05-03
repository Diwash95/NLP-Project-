#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install transformers')


# In[ ]:


# get transformers then bert along with other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install transformers')
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


# In[ ]:


#get dataset 
dataset = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')
dataset.head()


# In[ ]:


#delete unnecessary column
del dataset["version"]


# In[ ]:


# get ready to save only required columns into a new csv file
columns = ["text","question","answer"]

Final_list = []
for index, row in dataset.iterrows():
    for i in range(len(row["data"]["questions"])):
        Temporary = []
#append data to respective columns then compile
        Temporary.append(row["data"]["story"])
        Temporary.append(row["data"]["questions"][i]["input_text"])
        Temporary.append(row["data"]["answers"][i]["input_text"])
        Final_list.append(Temporary)
# store compiled data frame
new = pd.DataFrame(Final_list, columns=columns) 


# In[ ]:


#store processed data to new csv file
new.to_csv("new_data.csv", index=False)


# In[ ]:


# read dataset
data = pd.read_csv("new_data.csv")
data.head()


# In[ ]:


print("total questions & answers: ", len(data))


# BERT

# In[ ]:


#get the model and tokenizer 
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# In[ ]:


random_num = np.random.randint(0,len(data))
question = data["question"][random_num]
text = data["text"][random_num]


# In[ ]:


# tokenize question and text
input_ids = tokenizer.encode(question, text)


# In[ ]:


tokens = tokenizer.convert_ids_to_tokens(input_ids)

for token, id in zip(tokens, input_ids):
    print('{:8}{:8,}'.format(token,id))


# In[ ]:


#segmenting tokens in question and text 
idx = input_ids.index(tokenizer.sep_token_id)
print(idx)

#frequency of tokens in segment question
segment_question = idx+1
print(segment_question)

#frequency of tokens in segment text
segment_text = len(input_ids) - segment_question
print(segment_text)

segment_ids = [0]*segment_question + [1]*segment_text
print(segment_ids)

assert len(segment_ids) == len(input_ids)


# In[ ]:



# Feed the data to the model
output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))


# In[ ]:


#generate start and end scores of tokens in answer
start = torch.argmax(output.start_logits)
end = torch.argmax(output.end_logits)


# In[ ]:


if end >= start:
    answer = " ".join(tokens[start:end+1])
else:
    print("Try another question!")


# In[ ]:



start_scores = output.start_logits.detach().numpy().flatten()
end_scores = output.end_logits.detach().numpy().flatten()

token_labels = []
for i, token in enumerate(tokens):
    token_labels.append("{}-{}".format(token,i))


# In[ ]:



plt.rcParams["figure.figsize"] = (20,10)
ax = sns.barplot(x=token_labels[:80], y=start_scores[:80], ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
ax.grid(True)
plt.title("#first 100 tokensStart word scores")
plt.show()


# In[ ]:



plt.rcParams["figure.figsize"] = (20,10)
ax = sns.barplot(x=token_labels[-80:], y=start_scores[-80:], ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
ax.grid(True)
plt.title("last 100 tokensStart word scores")
plt.show()


# In[ ]:


answer = tokens[start]

for i in range(start+1, end+1):
    if tokens[i][0:2] == "##":
        answer += tokens[i][2:]
    else:
        answer += " " + tokens[i]


# In[ ]:


def questionanswer(question, text):
    
    #tokenize question and text in ids as a pair
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)

    #number of tokens in segment A - question
    segment_question = sep_idx+1

    #number of tokens in segment B - text
    segment_answer = len(input_ids) - segment_question
    
    #list of 0s and 1s
    segment_ids = [0]*segment_question + [1]*segment_answer
    
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
    #compile the answer
    start = torch.argmax(output.start_logits)
    end = torch.argmax(output.end_logits)

    if end >= start:
        answer = tokens[start]
        for i in range(start+1, end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
                
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    print("\nAnswer:\n{}".format(answer.capitalize()))


# In[ ]:


text = """Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is an entrepreneur, investor, and business magnate. He is the founder, CEO, and Chief Engineer at SpaceX; early-stage investor, CEO, and Product Architect of Tesla, Inc.; founder of The Boring Company; and co-founder of Neuralink and OpenAI. With an estimated net worth of around US$273 billion as of April 2022,[6] Musk is the wealthiest person in the world according to both the Bloomberg Billionaires Index and the Forbes real-time billionaires list.[7][8] Musk was born to a Canadian mother and White South African father, and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada at age 17 to avoid conscription. He was enrolled at Queen's University and transferred to the University of Pennsylvania two years later, where he received a bachelor's degree in economics and physics. He moved to California in 1995 to attend Stanford University but decided instead to pursue a business career, co-founding the web software company Zip2 with his brother Kimbal. The startup was acquired by Compaq for $307 million in 1999. The same year, Musk co-founded online bank X.com, which merged with Confinity in 2000 to form PayPal. The company was bought by eBay in 2002 for $1.5 billion."""
question = "Where did elon musk attend?"
print(question)
questionanswer(question, text)

