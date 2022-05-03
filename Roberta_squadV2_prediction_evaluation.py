#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('mkdir dataset_dir && cd dataset_dir')
get_ipython().system('wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json')
get_ipython().system('wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json')


# In[ ]:


import os
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer, squad_convert_examples_to_features)
from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits


# In[27]:


get_ipython().system('wget https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ -O evaluate-v2.py')


# In[ ]:


model_name = "deepset/roberta-base-squad2"

# Config
n_best_size = 1
max_answer_length = 30
do_lower_case = True
null_score_diff_threshold = 0.0

def to_list(tensor):
    return tensor.tolist()

config = RobertaConfig.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
model = RobertaForQuestionAnswering.from_pretrained(model_name, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

processor = SquadV2Processor()

def get_squad_example(question_text, context_text):
    return SquadExample(
        qas_id=str(0),
        question_text=question_text,
        context_text=context_text,
        answer_text=None,
        start_position_character=None,
        title="Predict",
        is_impossible=False,
        answers=None,
    )

def run_prediction(examples):
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

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]
            # print(example_indices)

            outputs = model(**inputs)
            # print(outputs)

            for i, example_index in enumerate(example_indices):
                # print(example_index)
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                # output = [to_list(output[i]) for output in outputs]
                # output = [list(map(int, output[i])) for output in outputs]
                # print(type(outputs[i]))

                start_logits = outputs.start_logits[i].tolist()
                end_logits = outputs.end_logits[i].tolist()
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    return predictions


# In[ ]:


def get_squad_example(question_texts, context_text):
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
    return examples


# In[ ]:


context = "New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean. It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million. New Zealand's capital city is Wellington, and its most populous city is Auckland."
questions = ["How many people live in New Zealand?", 
             "What's the largest city?"]

examples = get_squad_example(questions, context)

# Run method
predictions = run_prediction(examples)
# Print results
for key in predictions.keys():
  print(predictions[key])


# In[ ]:


#run inference on test data, either from the dev dataset or specific example from the website

test_data = processor.get_dev_examples("/content")
predictions = run_prediction(test_data)


# In[24]:


import json
with open("/content/dev-v2.0.json") as f:
  dataset_json = json.load(f)
  dataset = dataset_json["data"]

qid_to_has_ans = {}
for article in dataset:
  for p in article["paragraphs"]:
    for qa in p["qas"]:
      print(qa["answers"])
      # print(qa["id"], qa["answers"]["text"])
      # qid_to_has_ans[qa["id"]] = bool(qa["answers"]["text"])


# In[28]:


get_ipython().system('python /content/evaluate-v2.py /content/dev-v2.0.json /content/predictions.json -p /content')


# In[ ]:




