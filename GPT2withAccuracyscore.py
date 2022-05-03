#!/usr/bin/env python
# coding: utf-8

# In[47]:


get_ipython().system('git clone https://github.com/huggingface/transformers && cd transformers && git checkout a3085020ed0d81d4903c50967687192e3101e770 ')


# In[48]:


get_ipython().system('pip install ./transformers')
get_ipython().system('pip install tensorboardX')


# In[49]:


get_ipython().system('mkdir dataset && cd dataset && wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json && wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json')


# In[50]:


import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits
def intoList(tensor):
    return tensor.detach().cpu().tolist()
# for running predictions on given models
def run_prediction(model, model_name, tokenizer, question_texts, context_text):
    processor = SquadV2Processor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    # setting configuration
    n_best_size = 1
    max_answer_length = 30
    do_lower_case = True
    null_score_diff_threshold = 0.0
    """Setup function to compute predictions"""
    examples = []
    for i, question_text in enumerate(question_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None,
        )
        examples.append(example)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )
    sampler = SequentialSampler(dataset)
    Dataloader = DataLoader(dataset, sampler=sampler, batch_size=10)
    all_results = []
    for batch in Dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]
            outputs = model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [intoList(output[i]) for output in outputs]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    Pred_File = model_name + "_predictions.json"
    nbest_file =  model_name + "_nbest_predictions.json"
    outputnulllogoddsfile =  model_name + "_null_predictions.json"
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        Pred_File,
        nbest_file,
        outputnulllogoddsfile,
        False,  
        True,  
        null_score_diff_threshold,
        tokenizer,
    )
    return predictions


# In[51]:


# gpt2 model
def runGpt2(model, question_texts, context_text):
    predictions = []    
    for question in question_texts:
        prediction = model(question=question, context=context_text)
        predictions.append(prediction['answer'])        
    return predictions


# In[52]:


from transformers import pipeline
# for using pretrained gpt2 model
gpt2_model = pipeline('question-answering')


# In[53]:


# given dataset to test model
corpus = "Musk was born to a Canadian mother and White South African father, and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada at age 17 to avoid conscription. He was enrolled at Queen's University and transferred to the University of Pennsylvania two years later, where he received a bachelor's degree in economics and physics. He moved to California in 1995 to attend Stanford University but decided instead to pursue a business career, co-founding the web software company Zip2 with his brother Kimbal. The startup was acquired by Compaq for $307 million in 1999. The same year, Musk co-founded online bank X.com, which merged with Confinity in 2000 to form PayPal. The company was bought by eBay in 2002 for $1.5 billion."
query = ["What is startup by elon musk?"]


# In[54]:


Gpt2output = runGpt2(gpt2_model, query, corpus)


# In[55]:


print('question: ',*query, sep=" ") 
print('Answer:   ',*Gpt2output, sep=" ") 


# ACCURACY

# In[56]:


import json
from sklearn.metrics import accuracy_score
with open('dataset/dev-v2.0.json', 'r') as fp:
    Testx = json.load(fp)
Contxts = []
for data in Testx['data']:
    for paragraph in data['paragraphs']:
        context = paragraph['context']
        questions = []
        answers = []        
        for qas in paragraph['qas']:
            question = qas['question']
            q_answers = qas['answers']
            if len(q_answers) > 0:
                questions.append(question)
                answers.append(q_answers[0]['text'])
            else:
                p_answers = qas['plausible_answers']
                if len(p_answers) > 0:
                    questions.append(question)
                    answers.append(p_answers[0]['text'])
        Contxts.append({'context': context, 'questions':questions, 'answers': answers})


# In[57]:


# get x and y predicts
def Testx_prediction_gpt2(model, Contxts):
    y_predict =[]
    y_test =[]
    max_record = 0
    # limiting the records 
    for Contxt in Contxts[0:2]:
        print(max_record)
        predictions = runGpt2(model, Contxt['questions'], Contxt['context'])
        # answers
        y_test.extend(Contxt['answers'])
        # predictions
        for prediction in predictions:
            y_predict.append(prediction)            
        max_record += 1      
    return y_predict, y_test


# In[ ]:


y_predict, y_test = Testx_prediction_gpt2(gpt2_model, Contxts)
gpt2_accuracy = accuracy_score(y_test, y_predict)
print("Accuracy of GPT-2 is:",gpt2_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:




