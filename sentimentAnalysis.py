# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:26:25 2024

@author: jaack
"""

# Sentiment Analysis

from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import json


# Roberta transformer model from HuggingFace.co

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# load file

file_path = 'C:/Users/jaack/Desktop/Handmade_Products.jsonl'
reviews = []

with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        if 'text' in data:
            reviews.append(data['text'])
            
max_reviews = 100

limited_reviews = reviews[:max_reviews]


results = []

for review in limited_reviews:
    # Tokenize and truncate the review
    encoded_review = tokenizer(review, truncation=True, max_length=512, return_tensors='pt')
    # Perform sentiment analysis
    result = pipe(review, truncation=True, max_length=512)
    results.append(result)
    
for i, review in enumerate(reviews):
    print(f"Review: {review} \n Sentiment: {results[i]}\n")
