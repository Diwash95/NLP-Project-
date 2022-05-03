import streamlit as st
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('NLP - Project : Question and Answering ')

def model_question(model,context,question):
    if model == 'BERT':
        # write the BERT file
        return answer 
    elif model == 'ALBERT':
        # write the ALBERT file 
        return answer 
    elif model == 'GPT2':
        # write the GPT2 file 
        return answer 
    elif model == 'Roberta_Squad':
        # write the Roberta_Squad file 
        return answer 

model = st.radio(
     "Select the model to run the Question and Answer : ",
     ('BERT', 'ALBERT', 'GPT2','Roberta_Squad'))
if model == 'BERT':
     st.write('BERT model is selected')
elif model == 'ALBERT':
    st.write('ALBERT model is selected')
elif model == 'GPT2':
     st.write("GPT2 model is selected")
elif model == 'Roberta_Squad':
     st.write("Roberta_Squad model is selected")
else:
    st.write("Please select the model")

para_txt = st.text_area('Enter the Context : ',)
print(para_txt)
if len(para_txt) >0:
    st.write('The Context is Entered')
else:
    st.write('Context not Entered')

ques = st.text_input('Enter the Question : ')
if len(ques) >0:
    st.write('Entered Question : ',ques)
else:
    st.write('Question not given ')

ans=model_question(model,para_txt,ques)

st.write(ans)# The response answer is reflected in the webpage.